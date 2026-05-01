#!/usr/bin/env python3
from __future__ import annotations

import os
import re
import subprocess
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import requests


def resolve_server_python(user_specified: str) -> str:
    if user_specified.strip():
        return user_specified.strip()
    return sys.executable


def normalize_token(token: str) -> str:
    text = token.strip().lower()
    text = text.replace("ġ", "")
    return re.sub(r"[^a-z]", "", text)


def build_binary_prompt(problem_text: str, code_text: str) -> str:
    return (
        "You are an expert judge for Operations Research optimization code.\n"
        "Given a problem and a candidate solution code, decide if the code is "
        "correct and likely to solve the problem as intended.\n"
        "Respond with exactly one word: Yes or No.\n\n"
        "[Problem]\n"
        f"{problem_text}\n\n"
        "[Candidate Code]\n"
        f"{code_text}\n\n"
        "Is this candidate code correct for the problem?\n"
        "Answer:"
    )


@dataclass
class JudgeResult:
    label: str
    yes_prob: float
    no_prob: float
    raw_text: str
    top_logprobs: Dict[str, float]


class VLLMService:
    def __init__(
        self,
        *,
        model_path: str,
        host: str,
        port: int,
        gpu_devices: str,
        tensor_parallel_size: int,
        max_model_len: int,
        trust_remote_code: bool,
        start_server: bool,
        server_python: str,
        startup_check_endpoints: List[str],
        startup_log_file: str,
        startup_log_tail_lines: int,
    ):
        self.model_path = model_path
        self.host = host
        self.port = port
        self.gpu_devices = gpu_devices
        self.tensor_parallel_size = tensor_parallel_size
        self.max_model_len = max_model_len
        self.trust_remote_code = trust_remote_code
        self.start_server = start_server
        self.server_python = server_python
        self.startup_check_endpoints = startup_check_endpoints
        self.startup_log_file = startup_log_file
        self.startup_log_tail_lines = startup_log_tail_lines
        self.proc: Optional[subprocess.Popen] = None
        self._log_fh = None
        self._reader_thread: Optional[threading.Thread] = None
        self._log_tail: deque[str] = deque(maxlen=startup_log_tail_lines)

    @property
    def base_http(self) -> str:
        return f"http://{self.host}:{self.port}"

    @property
    def base_v1(self) -> str:
        return f"{self.base_http}/v1"

    def start(self) -> None:
        if not self.start_server:
            return

        env = os.environ.copy()
        if self.gpu_devices.strip():
            env["CUDA_VISIBLE_DEVICES"] = self.gpu_devices
        env.setdefault("VLLM_USE_MULTIPROCESSING_SPAWN", "1")

        cmd = [
            self.server_python,
            "-m",
            "vllm.entrypoints.openai.api_server",
            "--model",
            self.model_path,
            "--host",
            self.host,
            "--port",
            str(self.port),
            "--tensor-parallel-size",
            str(self.tensor_parallel_size),
            "--max-model-len",
            str(self.max_model_len),
        ]
        if self.trust_remote_code:
            cmd.append("--trust-remote-code")

        os.makedirs(os.path.dirname(self.startup_log_file), exist_ok=True)
        self._log_fh = open(self.startup_log_file, "w", encoding="utf-8")
        self.proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )
        self._reader_thread = threading.Thread(target=self._drain_stdout, daemon=True)
        self._reader_thread.start()

    def _drain_stdout(self) -> None:
        if self.proc is None or self.proc.stdout is None:
            return
        try:
            for line in self.proc.stdout:
                clean = line.rstrip("\n")
                self._log_tail.append(clean)
                if self._log_fh is not None:
                    self._log_fh.write(clean + "\n")
                    self._log_fh.flush()
        except Exception:
            return

    def _tail_text(self) -> str:
        if not self._log_tail:
            return "<no stdout captured>"
        return "\n".join(self._log_tail)

    def _probe_ready(self) -> Tuple[bool, Dict[str, str]]:
        statuses: Dict[str, str] = {}
        for endpoint in self.startup_check_endpoints:
            url = f"{self.base_http}{endpoint}"
            try:
                response = requests.get(url, timeout=3)
                statuses[endpoint] = str(response.status_code)
                if response.status_code == 200:
                    return True, statuses
            except Exception as exc:  # noqa: BLE001
                statuses[endpoint] = f"ERR:{exc.__class__.__name__}"
        return False, statuses

    def wait_ready(self, timeout_sec: int = 300) -> None:
        start = time.time()
        latest_statuses: Dict[str, str] = {}
        while time.time() - start < timeout_sec:
            ready, statuses = self._probe_ready()
            latest_statuses = statuses
            if ready:
                return
            if self.proc is not None and self.proc.poll() is not None:
                raise RuntimeError(
                    "vLLM exited before readiness probe succeeded. "
                    f"Statuses: {latest_statuses}\nLast logs:\n{self._tail_text()}"
                )
            time.sleep(2)
        raise TimeoutError(
            "Timed out waiting for vLLM readiness. "
            f"Statuses: {latest_statuses}\nLast logs:\n{self._tail_text()}"
        )

    def stop(self) -> None:
        if self.proc is not None and self.proc.poll() is None:
            try:
                self.proc.terminate()
                self.proc.wait(timeout=10)
            except Exception:
                try:
                    self.proc.kill()
                except Exception:
                    pass
        if self._log_fh is not None:
            self._log_fh.close()
            self._log_fh = None


class PRMDecider:
    def __init__(
        self,
        *,
        base_v1: str,
        model_id: str,
        api_key: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
        logprobs_k: int,
        request_timeout: int,
    ):
        self.base_v1 = base_v1.rstrip("/")
        self.model_id = model_id
        self.api_key = api_key
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.logprobs_k = logprobs_k
        self.request_timeout = request_timeout

    def judge(self, problem_text: str, code_text: str) -> JudgeResult:
        prompt = build_binary_prompt(problem_text, code_text)
        response = requests.post(
            f"{self.base_v1}/completions",
            json={
                "model": self.model_id,
                "prompt": prompt,
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
                "top_p": self.top_p,
                "logprobs": self.logprobs_k,
            },
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            timeout=self.request_timeout,
        )
        response.raise_for_status()
        payload = response.json()["choices"][0]
        raw_text = (payload.get("text") or "").strip()

        top_logprobs: Dict[str, float] = {}
        yes_prob = 0.0
        no_prob = 0.0
        logprobs_obj = payload.get("logprobs") or {}
        top_logprob_list = logprobs_obj.get("top_logprobs") or []
        if top_logprob_list and isinstance(top_logprob_list[0], dict):
            for token, logprob in top_logprob_list[0].items():
                try:
                    value = float(logprob)
                except Exception:
                    continue
                top_logprobs[token] = value
                normalized = normalize_token(token)
                if normalized == "yes":
                    yes_prob = max(yes_prob, pow(2.718281828459045, value))
                elif normalized == "no":
                    no_prob = max(no_prob, pow(2.718281828459045, value))

        raw_first = normalize_token(raw_text.split()[0] if raw_text else "")
        label = "UNKNOWN"
        if raw_first == "yes":
            label = "yes"
        elif raw_first == "no":
            label = "no"
        elif yes_prob > no_prob:
            label = "yes"
        elif no_prob > yes_prob:
            label = "no"

        return JudgeResult(
            label=label,
            yes_prob=yes_prob,
            no_prob=no_prob,
            raw_text=raw_text,
            top_logprobs=top_logprobs,
        )


def choose_by_rule(candidate_a: JudgeResult, candidate_b: JudgeResult) -> Tuple[str, str]:
    if candidate_a.label == "yes" and candidate_b.label == "yes":
        if candidate_a.yes_prob >= candidate_b.yes_prob:
            return "A", "both_yes_pick_higher_yes_prob"
        return "B", "both_yes_pick_higher_yes_prob"

    if candidate_a.label == "no" and candidate_b.label == "no":
        if candidate_a.no_prob <= candidate_b.no_prob:
            return "A", "both_no_pick_lower_no_prob"
        return "B", "both_no_pick_lower_no_prob"

    if candidate_a.label == "yes" and candidate_b.label != "yes":
        return "A", "one_yes_one_non_yes_pick_yes"
    if candidate_b.label == "yes" and candidate_a.label != "yes":
        return "B", "one_yes_one_non_yes_pick_yes"

    score_a = candidate_a.yes_prob - candidate_a.no_prob
    score_b = candidate_b.yes_prob - candidate_b.no_prob
    if score_a >= score_b:
        return "A", "fallback_margin_yes_minus_no"
    return "B", "fallback_margin_yes_minus_no"


def discover_model_id(base_v1: str, api_key: str, prefer: Optional[str] = None) -> str:
    response = requests.get(
        f"{base_v1.rstrip('/')}/models",
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=20,
    )
    response.raise_for_status()
    models = response.json().get("data", [])
    ids = [item.get("id") for item in models if item.get("id")]
    if not ids:
        raise RuntimeError("No model IDs returned by /v1/models.")
    if prefer:
        if prefer in ids:
            return prefer
        basename = os.path.basename(prefer.rstrip("/"))
        for model_id in ids:
            if model_id == basename or os.path.basename(model_id.rstrip("/")) == basename:
                return model_id
    return ids[0]
