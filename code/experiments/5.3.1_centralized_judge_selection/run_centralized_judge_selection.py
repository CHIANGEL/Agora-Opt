#!/usr/bin/env python3
from __future__ import annotations

import argparse
import atexit
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import requests

from prm_decider import (
    PRMDecider,
    VLLMService,
    choose_by_rule,
    discover_model_id,
    resolve_server_python,
)


MAX_MODEL_LEN = 32768
PRM_MAX_TOKENS = 1
PRM_TEMPERATURE = 0.0
PRM_TOP_P = 1.0
PRM_LOGPROBS_K = 20
JUDGE_MAX_TOKENS = 25000
DEFAULT_RELATIVE_ERROR_THRESHOLD = 0.05
DEFAULT_MODEL_PATH = (
    Path(__file__).resolve().parents[3] / "models" / "steporlm"
)


@dataclass(frozen=True)
class DatasetBundle:
    family_name: str
    display_name: str
    generation_a: Path
    evaluation_a: Path
    generation_b: Path
    evaluation_b: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run centralized judge-selection experiments from explicitly specified "
            "initial solution files."
        )
    )
    parser.add_argument(
        "--dataset-config",
        action="append",
        required=True,
        help=(
            "Dataset spec. Format: family|display_name|generation_a|evaluation_a|"
            "generation_b|evaluation_b. display_name may be omitted."
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(__file__).resolve().parent / "runs",
        help="Directory where experiment outputs are written.",
    )
    parser.add_argument(
        "--model-a-name",
        type=str,
        default="deepseek-chat",
        help="Model label for side A.",
    )
    parser.add_argument(
        "--model-b-name",
        type=str,
        default="gpt-4o",
        help="Model label for side B.",
    )
    parser.add_argument(
        "--experiments",
        nargs="+",
        choices=["prm", "gemini", "deepseek_v3", "both"],
        default=["both"],
        help="Which experiments to run.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Optional cap per dataset. 0 means full dataset.",
    )
    parser.add_argument(
        "--relative-error-threshold",
        type=float,
        default=DEFAULT_RELATIVE_ERROR_THRESHOLD,
        help="Relative error threshold used for correctness.",
    )

    parser.add_argument("--host", type=str, default="127.0.0.1", help="vLLM host.")
    parser.add_argument("--port", type=int, default=8001, help="vLLM port.")
    parser.add_argument(
        "--gpu-devices",
        type=str,
        default="0,1,2,3",
        help="CUDA_VISIBLE_DEVICES used when starting vLLM.",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=str(DEFAULT_MODEL_PATH),
        help="Local GenPRM model path used by vLLM.",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="",
        help="Optional request model id. Leave empty to auto-discover from vLLM.",
    )
    parser.add_argument(
        "--server-python",
        type=str,
        default="",
        help="Python executable used to start vLLM.",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default="local-prm",
        help="API key used for the local OpenAI-compatible vLLM endpoint.",
    )
    parser.add_argument(
        "--startup-timeout",
        type=int,
        default=300,
        help="Seconds to wait for vLLM readiness.",
    )
    parser.add_argument(
        "--request-timeout",
        type=int,
        default=120,
        help="Per-request timeout passed to the PRM judge client.",
    )
    parser.add_argument(
        "--no-start-server",
        action="store_true",
        help="Connect to an already-running vLLM server instead of starting one.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Pass --trust-remote-code when starting vLLM.",
    )
    parser.add_argument(
        "--judge-api-key",
        type=str,
        default="",
        help="Optional centralized-judge API key. Defaults to environment variables.",
    )
    parser.add_argument(
        "--judge-api-base-url",
        type=str,
        default="",
        help="Optional centralized-judge API base URL. Defaults to environment variables.",
    )
    parser.add_argument(
        "--judge-timeout",
        type=float,
        default=180.0,
        help="Per-request timeout for Gemini / DeepSeek-V3 judges.",
    )
    return parser.parse_args()


def parse_dataset_config(raw: str) -> DatasetBundle:
    parts = raw.split("|")
    if len(parts) == 5:
        family_name, generation_a, evaluation_a, generation_b, evaluation_b = parts
        display_name = family_name
    elif len(parts) == 6:
        family_name, display_name, generation_a, evaluation_a, generation_b, evaluation_b = parts
    else:
        raise ValueError(
            "Invalid --dataset-config. Expected 5 or 6 pipe-separated fields, "
            f"got {len(parts)}: {raw}"
        )
    return DatasetBundle(
        family_name=family_name,
        display_name=display_name,
        generation_a=Path(generation_a).expanduser().resolve(),
        evaluation_a=Path(evaluation_a).expanduser().resolve(),
        generation_b=Path(generation_b).expanduser().resolve(),
        evaluation_b=Path(evaluation_b).expanduser().resolve(),
    )


def ensure_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Required file not found: {path}")


def load_jsonl_by_id(path: Path, id_key: str = "id") -> Dict[Any, Dict[str, Any]]:
    rows: Dict[Any, Dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            obj = json.loads(line)
            row_id = obj.get(id_key)
            if row_id is None:
                row_id = f"line_{line_number}"
            rows[row_id] = obj
    return rows


def parse_numeric(value: Any) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric):
        return None
    return numeric


def is_correct(predicted: Any, ground_truth: Any, threshold: float) -> bool:
    pred = parse_numeric(predicted)
    truth = parse_numeric(ground_truth)
    if pred is None or truth is None:
        return False
    if truth == 0.0:
        return abs(pred - truth) <= threshold
    return abs(pred - truth) / abs(truth) <= threshold


def build_candidate_snapshot(model_name: str, result: Any, generated_code: str) -> Dict[str, Any]:
    return {
        "model": model_name,
        "result": result,
        "generated_code": generated_code,
    }


def build_debate_style_record(
    *,
    sample: Dict[str, Any],
    chosen_model: str,
    chosen_side: str,
    chosen_code: str,
    chosen_result: Any,
    decision_method: str,
    rule: str,
    judge_payload: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    record = {
        "id": sample["problem_id"],
        "problem_id": sample["problem_id"],
        "description": sample["problem_text"],
        "answer": sample["ground_truth"],
        "status": "decision_pick",
        "generated_code": chosen_code,
        "final_result": chosen_result,
        "chosen_model": chosen_model,
        "chosen_side": chosen_side,
        "decision_method": decision_method,
        "decision_rule": rule,
        "initial_A_result": sample["initial_A_result"],
        "initial_B_result": sample["initial_B_result"],
        "initial_A_code": sample["initial_answer"][sample["model_a_name"]],
        "initial_B_code": sample["initial_answer"][sample["model_b_name"]],
        "initial_A": build_candidate_snapshot(
            sample["model_a_name"],
            sample["initial_A_result"],
            sample["initial_answer"][sample["model_a_name"]],
        ),
        "initial_B": build_candidate_snapshot(
            sample["model_b_name"],
            sample["initial_B_result"],
            sample["initial_answer"][sample["model_b_name"]],
        ),
    }
    if judge_payload:
        record.update(judge_payload)
    return record


def build_combined_input_file(
    output_path: Path,
    bundle: DatasetBundle,
    *,
    model_a_name: str,
    model_b_name: str,
    max_samples: int,
) -> Dict[str, Any]:
    generation_a = load_jsonl_by_id(bundle.generation_a)
    generation_b = load_jsonl_by_id(bundle.generation_b)
    evaluation_a = load_jsonl_by_id(bundle.evaluation_a)
    evaluation_b = load_jsonl_by_id(bundle.evaluation_b)

    common_ids = sorted(set(generation_a) & set(generation_b) & set(evaluation_a) & set(evaluation_b))
    if max_samples > 0:
        common_ids = common_ids[:max_samples]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    a_correct = 0
    b_correct = 0
    oracle_correct = 0

    with output_path.open("w", encoding="utf-8") as handle:
        for problem_id in common_ids:
            generation_row_a = generation_a[problem_id]
            generation_row_b = generation_b[problem_id]
            evaluation_row_a = evaluation_a[problem_id]
            evaluation_row_b = evaluation_b[problem_id]

            ground_truth = evaluation_row_a.get("ground_truth", evaluation_row_b.get("ground_truth"))
            result_a = evaluation_row_a.get("predicted_objective")
            result_b = evaluation_row_b.get("predicted_objective")
            problem_text = (
                generation_row_a.get("description")
                or generation_row_b.get("description")
                or ""
            )
            a_ok = bool(evaluation_row_a.get("is_correct"))
            b_ok = bool(evaluation_row_b.get("is_correct"))
            a_correct += int(a_ok)
            b_correct += int(b_ok)
            oracle_correct += int(a_ok or b_ok)

            merged = {
                "problem_id": problem_id,
                "problem_text": problem_text,
                "ground_truth": ground_truth,
                "initial_A_result": result_a,
                "initial_B_result": result_b,
                "initial_answer": {
                    model_a_name: generation_row_a.get("generated_code", ""),
                    model_b_name: generation_row_b.get("generated_code", ""),
                },
                "model_a_name": model_a_name,
                "model_b_name": model_b_name,
                "meta": {
                    "generation_a": str(bundle.generation_a),
                    "evaluation_a": str(bundle.evaluation_a),
                    "generation_b": str(bundle.generation_b),
                    "evaluation_b": str(bundle.evaluation_b),
                    "initial_A_is_correct": a_ok,
                    "initial_B_is_correct": b_ok,
                },
            }
            handle.write(json.dumps(merged, ensure_ascii=False) + "\n")

    total = len(common_ids)
    return {
        "total": total,
        "initial_A_correct_count": a_correct,
        "initial_B_correct_count": b_correct,
        "oracle_best_of_two_correct_count": oracle_correct,
        "initial_A_accuracy": (a_correct / total) if total else None,
        "initial_B_accuracy": (b_correct / total) if total else None,
        "oracle_best_of_two_accuracy": (oracle_correct / total) if total else None,
    }


def _has_result(value: Any) -> bool:
    return parse_numeric(value) is not None


def choose_with_fallback(
    *,
    result_a: Any,
    result_b: Any,
    decider: PRMDecider,
    problem_text: str,
    code_a: str,
    code_b: str,
) -> Tuple[str, str, Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    has_a = _has_result(result_a)
    has_b = _has_result(result_b)

    if has_a and not has_b:
        return "A", "fallback_only_A_has_result", None, None
    if has_b and not has_a:
        return "B", "fallback_only_B_has_result", None, None

    availability_state = "both_have_result" if has_a and has_b else "neither_has_result"
    judge_a = decider.judge(problem_text, code_a)
    judge_b = decider.judge(problem_text, code_b)
    chosen_side, judge_rule = choose_by_rule(judge_a, judge_b)
    rule = f"{availability_state}:{judge_rule}"

    payload_a = {
        "label": judge_a.label,
        "yes_prob": judge_a.yes_prob,
        "no_prob": judge_a.no_prob,
        "raw_output": judge_a.raw_text,
        "top_logprobs": judge_a.top_logprobs,
    }
    payload_b = {
        "label": judge_b.label,
        "yes_prob": judge_b.yes_prob,
        "no_prob": judge_b.no_prob,
        "raw_output": judge_b.raw_text,
        "top_logprobs": judge_b.top_logprobs,
    }
    return chosen_side, rule, payload_a, payload_b


def resolve_judge_credentials(args: argparse.Namespace) -> Tuple[str, str]:
    api_key = (
        args.judge_api_key
        or os.getenv("LLM_API_KEY")
        or os.getenv("OPENAI_API_KEY")
        or os.getenv("API_KEY")
    )
    base_url = (
        args.judge_api_base_url
        or os.getenv("LLM_API_BASE_URL")
        or os.getenv("OPENAI_BASE_URL")
        or os.getenv("API_URL")
    )
    if not api_key:
        raise RuntimeError("Missing centralized judge API key.")
    if not base_url:
        raise RuntimeError("Missing centralized judge API base URL.")
    return api_key, base_url.rstrip("/")


def call_chat_judge(
    *,
    prompt: str,
    model_name: str,
    args: argparse.Namespace,
) -> str:
    api_key, base_url = resolve_judge_credentials(args)
    response = requests.post(
        f"{base_url}/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
            "max_tokens": JUDGE_MAX_TOKENS,
        },
        timeout=args.judge_timeout,
    )
    response.raise_for_status()
    payload = response.json()
    return (((payload.get("choices") or [{}])[0].get("message") or {}).get("content") or "").strip()


def llm_pick_judge(
    *,
    sample: Dict[str, Any],
    judge_model: str,
    args: argparse.Namespace,
) -> Tuple[str, str, str]:
    result_a = sample["initial_A_result"]
    result_b = sample["initial_B_result"]
    has_a = _has_result(result_a)
    has_b = _has_result(result_b)

    if has_a and not has_b:
        return "A", "fallback_only_A_has_result", ""
    if has_b and not has_a:
        return "B", "fallback_only_B_has_result", ""

    availability_state = "both_have_result" if has_a and has_b else "neither_has_result"
    prompt = (
        "You are an expert judge evaluating two Python code solutions for an "
        "operations research optimization problem.\n\n"
        "**Problem Description:**\n"
        f"{sample['problem_text']}\n\n"
        f"**Solution A ({sample['model_a_name']}):**\n```python\n"
        f"{sample['initial_answer'][sample['model_a_name']]}\n```\n"
        f"Solution A execution result: {result_a}\n\n"
        f"**Solution B ({sample['model_b_name']}):**\n```python\n"
        f"{sample['initial_answer'][sample['model_b_name']]}\n```\n"
        f"Solution B execution result: {result_b}\n\n"
        "Analyze both solutions carefully. Consider:\n"
        "1. Whether the mathematical formulation correctly models the problem\n"
        "2. Whether the implementation is correct\n"
        "3. The execution results\n"
        "4. Which solution is more likely to produce the correct optimal answer\n\n"
        "You MUST respond with ONLY a single letter: \"A\" or \"B\"."
    )
    raw_response = call_chat_judge(prompt=prompt, model_name=judge_model, args=args)
    upper = raw_response.strip().upper()
    if "A" in upper and "B" not in upper:
        return "A", f"{availability_state}:{judge_model}_chose_A", raw_response
    if "B" in upper and "A" not in upper:
        return "B", f"{availability_state}:{judge_model}_chose_B", raw_response
    if upper.startswith("A"):
        return "A", f"{availability_state}:{judge_model}_chose_A", raw_response
    if upper.startswith("B"):
        return "B", f"{availability_state}:{judge_model}_chose_B", raw_response
    return "A", f"{availability_state}:{judge_model}_unparseable", raw_response


def run_prm_decision(
    *,
    input_file: Path,
    output_dir: Path,
    decider: PRMDecider,
    model_a_name: str,
    model_b_name: str,
) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    decisions_path = output_dir / "decisions.jsonl"
    debate_path = output_dir / "debate_format_results.jsonl"

    total = 0
    ok = 0
    errors = 0
    fallback_count = 0

    with input_file.open("r", encoding="utf-8") as source, decisions_path.open("w", encoding="utf-8") as decisions, debate_path.open("w", encoding="utf-8") as debate:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            total += 1
            record: Dict[str, Any] = {"line": line_number, "status": "ok"}
            try:
                sample = json.loads(line)
                code_a = sample["initial_answer"][model_a_name]
                code_b = sample["initial_answer"][model_b_name]
                chosen_side, rule, judge_a, judge_b = choose_with_fallback(
                    result_a=sample["initial_A_result"],
                    result_b=sample["initial_B_result"],
                    decider=decider,
                    problem_text=sample["problem_text"],
                    code_a=code_a,
                    code_b=code_b,
                )
                if rule.startswith("fallback_"):
                    fallback_count += 1

                chosen_model = model_a_name if chosen_side == "A" else model_b_name
                chosen_code = code_a if chosen_side == "A" else code_b
                chosen_result = sample["initial_A_result"] if chosen_side == "A" else sample["initial_B_result"]

                record.update(
                    {
                        "problem_id": sample["problem_id"],
                        "description": sample["problem_text"],
                        "chosen_model": chosen_model,
                        "chosen_side": chosen_side,
                        "generated_code": chosen_code,
                        "final_result": chosen_result,
                        "decision": {
                            "chosen_model": chosen_model,
                            "chosen_side": chosen_side,
                            "generated_code": chosen_code,
                            "final_result": chosen_result,
                            "rule": rule,
                        },
                        "initial_candidates": {
                            "A": build_candidate_snapshot(model_a_name, sample["initial_A_result"], code_a),
                            "B": build_candidate_snapshot(model_b_name, sample["initial_B_result"], code_b),
                        },
                    }
                )
                if judge_a is not None:
                    record["judge_A"] = judge_a
                if judge_b is not None:
                    record["judge_B"] = judge_b
                debate.write(
                    json.dumps(
                        build_debate_style_record(
                            sample=sample,
                            chosen_model=chosen_model,
                            chosen_side=chosen_side,
                            chosen_code=chosen_code,
                            chosen_result=chosen_result,
                            decision_method="prm_with_fallback",
                            rule=rule,
                            judge_payload={"judge_A": judge_a, "judge_B": judge_b},
                        ),
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                ok += 1
            except Exception as exc:  # noqa: BLE001
                record["status"] = "error"
                record["error"] = str(exc)
                errors += 1
            decisions.write(json.dumps(record, ensure_ascii=False) + "\n")

    summary = {
        "method": "prm_with_fallback",
        "input_file": str(input_file),
        "output_jsonl": str(decisions_path),
        "debate_format_jsonl": str(debate_path),
        "total": total,
        "ok": ok,
        "errors": errors,
        "fallback_count": fallback_count,
    }
    write_json(output_dir / "decision_summary.json", summary)
    return summary


def run_llm_pick_decision(
    *,
    input_file: Path,
    output_dir: Path,
    judge_model: str,
    method_label: str,
    model_a_name: str,
    model_b_name: str,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    decisions_path = output_dir / "decisions.jsonl"
    debate_path = output_dir / "debate_format_results.jsonl"

    total = 0
    ok = 0
    errors = 0
    fallback_count = 0

    with input_file.open("r", encoding="utf-8") as source, decisions_path.open("w", encoding="utf-8") as decisions, debate_path.open("w", encoding="utf-8") as debate:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            total += 1
            record: Dict[str, Any] = {"line": line_number, "status": "ok"}
            try:
                sample = json.loads(line)
                chosen_side, rule, raw_response = llm_pick_judge(
                    sample=sample,
                    judge_model=judge_model,
                    args=args,
                )
                if rule.startswith("fallback_"):
                    fallback_count += 1

                code_a = sample["initial_answer"][model_a_name]
                code_b = sample["initial_answer"][model_b_name]
                chosen_model = model_a_name if chosen_side == "A" else model_b_name
                chosen_code = code_a if chosen_side == "A" else code_b
                chosen_result = sample["initial_A_result"] if chosen_side == "A" else sample["initial_B_result"]

                record.update(
                    {
                        "problem_id": sample["problem_id"],
                        "description": sample["problem_text"],
                        "judge_model": judge_model,
                        "judge_raw_response": raw_response,
                        "chosen_model": chosen_model,
                        "chosen_side": chosen_side,
                        "generated_code": chosen_code,
                        "final_result": chosen_result,
                        "decision": {
                            "chosen_model": chosen_model,
                            "chosen_side": chosen_side,
                            "generated_code": chosen_code,
                            "final_result": chosen_result,
                            "rule": rule,
                        },
                        "initial_candidates": {
                            "A": build_candidate_snapshot(model_a_name, sample["initial_A_result"], code_a),
                            "B": build_candidate_snapshot(model_b_name, sample["initial_B_result"], code_b),
                        },
                    }
                )
                debate.write(
                    json.dumps(
                        build_debate_style_record(
                            sample=sample,
                            chosen_model=chosen_model,
                            chosen_side=chosen_side,
                            chosen_code=chosen_code,
                            chosen_result=chosen_result,
                            decision_method=method_label,
                            rule=rule,
                            judge_payload={
                                "judge_model": judge_model,
                                "judge_raw_response": raw_response,
                            },
                        ),
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                ok += 1
            except Exception as exc:  # noqa: BLE001
                record["status"] = "error"
                record["error"] = str(exc)
                errors += 1
            decisions.write(json.dumps(record, ensure_ascii=False) + "\n")

    summary = {
        "method": method_label,
        "judge_model": judge_model,
        "input_file": str(input_file),
        "output_jsonl": str(decisions_path),
        "debate_format_jsonl": str(debate_path),
        "total": total,
        "ok": ok,
        "errors": errors,
        "fallback_count": fallback_count,
    }
    write_json(output_dir / "decision_summary.json", summary)
    return summary


def evaluate_decisions(
    *,
    combined_file: Path,
    decisions_file: Path,
    relative_error_threshold: float,
    detail_output_file: Path,
) -> Dict[str, Any]:
    combined_by_pid = load_jsonl_by_id(combined_file, id_key="problem_id")
    total_decisions = 0
    joined_rows = 0
    pick_correct = 0
    xor_subset_total = 0
    pick_correct_in_xor = 0
    unmatched: List[Any] = []

    detail_output_file.parent.mkdir(parents=True, exist_ok=True)
    with decisions_file.open("r", encoding="utf-8") as source, detail_output_file.open("w", encoding="utf-8") as details:
        for line in source:
            if not line.strip():
                continue
            total_decisions += 1
            decision_row = json.loads(line)
            problem_id = decision_row.get("problem_id")
            if problem_id not in combined_by_pid:
                unmatched.append(problem_id)
                continue

            combined_row = combined_by_pid[problem_id]
            joined_rows += 1
            picked_result = decision_row.get("final_result")
            ground_truth = combined_row.get("ground_truth")
            a_result = combined_row.get("initial_A_result")
            b_result = combined_row.get("initial_B_result")
            picked_ok = is_correct(picked_result, ground_truth, relative_error_threshold)
            if picked_ok:
                pick_correct += 1

            a_ok = is_correct(a_result, ground_truth, relative_error_threshold)
            b_ok = is_correct(b_result, ground_truth, relative_error_threshold)
            if a_ok ^ b_ok:
                xor_subset_total += 1
                if picked_ok:
                    pick_correct_in_xor += 1

            details.write(
                json.dumps(
                    {
                        "problem_id": problem_id,
                        "description": combined_row.get("problem_text", ""),
                        "ground_truth": ground_truth,
                        "initial_A_result": a_result,
                        "initial_B_result": b_result,
                        "initial_A_correct": a_ok,
                        "initial_B_correct": b_ok,
                        "chosen_model": decision_row.get("chosen_model"),
                        "chosen_side": decision_row.get("chosen_side"),
                        "decision_rule": (decision_row.get("decision") or {}).get("rule"),
                        "picked_result": picked_result,
                        "picked_correct": picked_ok,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    return {
        "counts": {
            "total_decisions_rows": total_decisions,
            "joined_rows": joined_rows,
            "unmatched_problem_ids_count": len(unmatched),
        },
        "pick_accuracy": {
            "correct": pick_correct,
            "total": joined_rows,
            "accuracy": (pick_correct / joined_rows) if joined_rows else None,
        },
        "xor_one_correct_between_initial_A_B": {
            "subset_total": xor_subset_total,
            "pick_correct_count": pick_correct_in_xor,
            "pick_accuracy_in_subset": (
                (pick_correct_in_xor / xor_subset_total) if xor_subset_total else None
            ),
        },
        "unmatched_problem_ids_preview": unmatched[:20],
    }


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def tensor_parallel_size_from_devices(gpu_devices: str) -> int:
    devices = [item.strip() for item in gpu_devices.split(",") if item.strip()]
    return max(1, len(devices))


def start_service(args: argparse.Namespace, output_root: Path) -> Tuple[VLLMService, str]:
    if not args.model_path.strip():
        raise ValueError("--model-path is required when running the PRM experiment.")
    service = VLLMService(
        model_path=args.model_path,
        host=args.host,
        port=args.port,
        gpu_devices=args.gpu_devices,
        tensor_parallel_size=tensor_parallel_size_from_devices(args.gpu_devices),
        max_model_len=MAX_MODEL_LEN,
        trust_remote_code=args.trust_remote_code,
        start_server=not args.no_start_server,
        server_python=resolve_server_python(args.server_python),
        startup_check_endpoints=["/health", "/v1/models"],
        startup_log_file=str(output_root / "vllm_server.log"),
        startup_log_tail_lines=200,
    )
    service.start()
    service.wait_ready(timeout_sec=args.startup_timeout)
    atexit.register(service.stop)

    model_id = args.model_id.strip()
    if not model_id:
        model_id = discover_model_id(service.base_v1, args.api_key, prefer=args.model_path)
    return service, model_id


def build_decider(service: VLLMService, model_id: str, args: argparse.Namespace) -> PRMDecider:
    return PRMDecider(
        base_v1=service.base_v1,
        model_id=model_id,
        api_key=args.api_key,
        max_tokens=PRM_MAX_TOKENS,
        temperature=PRM_TEMPERATURE,
        top_p=PRM_TOP_P,
        logprobs_k=PRM_LOGPROBS_K,
        request_timeout=args.request_timeout,
    )


def run_single_experiment(
    *,
    experiment_name: str,
    method_label: str,
    bundles: Iterable[DatasetBundle],
    output_root: Path,
    args: argparse.Namespace,
    decider: Optional[PRMDecider] = None,
) -> Dict[str, Any]:
    exp_root = output_root / experiment_name
    exp_root.mkdir(parents=True, exist_ok=True)
    experiment_summary: Dict[str, Any] = {"method": method_label, "datasets": {}}
    table_rows: List[Dict[str, Any]] = []

    for bundle in bundles:
        dataset_dir = exp_root / bundle.family_name
        combined_file = dataset_dir / "combined_input.jsonl"
        baseline_summary = build_combined_input_file(
            output_path=combined_file,
            bundle=bundle,
            model_a_name=args.model_a_name,
            model_b_name=args.model_b_name,
            max_samples=args.max_samples,
        )

        if experiment_name == "prm_with_fallback":
            assert decider is not None
            decision_summary = run_prm_decision(
                input_file=combined_file,
                output_dir=dataset_dir,
                decider=decider,
                model_a_name=args.model_a_name,
                model_b_name=args.model_b_name,
            )
        elif experiment_name == "gemini_judge":
            decision_summary = run_llm_pick_decision(
                input_file=combined_file,
                output_dir=dataset_dir,
                judge_model="gemini-2.5-pro",
                method_label="gemini-2.5-pro",
                model_a_name=args.model_a_name,
                model_b_name=args.model_b_name,
                args=args,
            )
        elif experiment_name == "deepseek_v3_judge":
            decision_summary = run_llm_pick_decision(
                input_file=combined_file,
                output_dir=dataset_dir,
                judge_model="deepseek-v3",
                method_label="deepseek-v3",
                model_a_name=args.model_a_name,
                model_b_name=args.model_b_name,
                args=args,
            )
        else:
            raise ValueError(f"Unsupported experiment: {experiment_name}")

        eval_summary = evaluate_decisions(
            combined_file=combined_file,
            decisions_file=dataset_dir / "decisions.jsonl",
            relative_error_threshold=args.relative_error_threshold,
            detail_output_file=dataset_dir / "evaluation_details.jsonl",
        )
        dataset_summary = {
            "dataset_name": bundle.family_name,
            "display_name": bundle.display_name,
            "result_files": {
                "combined_input_jsonl": str(combined_file),
                "decisions_jsonl": str(dataset_dir / "decisions.jsonl"),
                "debate_format_results_jsonl": str(dataset_dir / "debate_format_results.jsonl"),
                "evaluation_details_jsonl": str(dataset_dir / "evaluation_details.jsonl"),
            },
            "source_files": {
                args.model_a_name: {
                    "generation": str(bundle.generation_a),
                    "evaluation_results": str(bundle.evaluation_a),
                },
                args.model_b_name: {
                    "generation": str(bundle.generation_b),
                    "evaluation_results": str(bundle.evaluation_b),
                },
            },
            "baseline_summary": baseline_summary,
            "decision_summary": decision_summary,
            "evaluation_summary": eval_summary,
        }
        write_json(dataset_dir / "experiment_summary.json", dataset_summary)
        experiment_summary["datasets"][bundle.family_name] = dataset_summary

        total = baseline_summary["total"]
        table_rows.append(
            {
                "dataset": bundle.display_name,
                "pick_correct": eval_summary["pick_accuracy"]["correct"],
                "initial_A_correct": baseline_summary["initial_A_correct_count"],
                "initial_B_correct": baseline_summary["initial_B_correct_count"],
                "total": total,
                "pick_accuracy": eval_summary["pick_accuracy"]["accuracy"],
                "initial_A_accuracy": baseline_summary["initial_A_accuracy"],
                "initial_B_accuracy": baseline_summary["initial_B_accuracy"],
            }
        )

    experiment_summary["table_rows"] = table_rows
    write_json(exp_root / "experiment_summary.json", experiment_summary)
    return experiment_summary


def main() -> None:
    args = parse_args()
    bundles = [parse_dataset_config(raw) for raw in args.dataset_config]
    for bundle in bundles:
        ensure_file(bundle.generation_a)
        ensure_file(bundle.evaluation_a)
        ensure_file(bundle.generation_b)
        ensure_file(bundle.evaluation_b)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_root = args.output_root.resolve() / timestamp
    output_root.mkdir(parents=True, exist_ok=True)

    experiments_to_run = set()
    for experiment in args.experiments:
        if experiment == "both":
            experiments_to_run.update(["prm", "gemini"])
        else:
            experiments_to_run.add(experiment)

    overall_summary: Dict[str, Any] = {
        "config": {
            "output_root": str(output_root),
            "experiments": sorted(experiments_to_run),
            "model_a_name": args.model_a_name,
            "model_b_name": args.model_b_name,
            "relative_error_threshold": args.relative_error_threshold,
            "max_samples": args.max_samples,
        },
        "datasets": [
            {
                "family_name": bundle.family_name,
                "display_name": bundle.display_name,
                "generation_a": str(bundle.generation_a),
                "evaluation_a": str(bundle.evaluation_a),
                "generation_b": str(bundle.generation_b),
                "evaluation_b": str(bundle.evaluation_b),
            }
            for bundle in bundles
        ],
        "experiments": {},
    }

    service: Optional[VLLMService] = None
    if "prm" in experiments_to_run:
        try:
            service, model_id = start_service(args, output_root)
            decider = build_decider(service, model_id, args)
            overall_summary["experiments"]["prm_with_fallback"] = run_single_experiment(
                experiment_name="prm_with_fallback",
                method_label="GenPRM (with fallback)",
                bundles=bundles,
                output_root=output_root,
                args=args,
                decider=decider,
            )
        finally:
            if service is not None and not args.no_start_server:
                service.stop()
                service = None

    if "gemini" in experiments_to_run:
        overall_summary["experiments"]["gemini_judge"] = run_single_experiment(
            experiment_name="gemini_judge",
            method_label="Gemini-2.5-pro",
            bundles=bundles,
            output_root=output_root,
            args=args,
        )

    if "deepseek_v3" in experiments_to_run:
        overall_summary["experiments"]["deepseek_v3_judge"] = run_single_experiment(
            experiment_name="deepseek_v3_judge",
            method_label="deepseek-v3",
            bundles=bundles,
            output_root=output_root,
            args=args,
        )

    write_json(output_root / "overall_summary.json", overall_summary)
    print(f"Output root: {output_root}")
    print(f"Summary: {output_root / 'overall_summary.json'}")


if __name__ == "__main__":
    main()
