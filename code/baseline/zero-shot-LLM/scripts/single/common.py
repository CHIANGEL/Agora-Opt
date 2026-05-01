"""Common utilities for single-run generation/evaluation scripts."""

from __future__ import annotations

import datetime as dt
import json
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any


SIMPLE_OR_PROMPT_TEMPLATE = (
	"You are an expert in operations research and mathematical optimization.\n"
	"Given the problem below, think it through and solve it with Python code.\n"
	"Use gurobipy when suitable, or another Python optimization library if needed.\n"
	"Please include runnable Python code and make sure the code prints the final numeric result.\n\n"
	"Problem:\n{problem}\n"
)


def now_ts_compact() -> str:
	"""Return current timestamp in a filename-friendly format."""
	return dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def build_optimization_prompt(problem_text: str) -> str:
	"""Embed raw problem text into a simple optimization prompt template."""
	return SIMPLE_OR_PROMPT_TEMPLATE.format(problem=problem_text)


def ensure_dir(path: Path) -> Path:
	"""Create directory if needed and return it."""
	path.mkdir(parents=True, exist_ok=True)
	return path


def safe_model_name(model: str) -> str:
	"""Normalize model string so it can be used as a directory name."""
	model = model.strip()
	return re.sub(r"[^a-zA-Z0-9._-]+", "_", model) or "unknown_model"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
	"""Read JSONL file into a list of JSON objects."""
	rows: list[dict[str, Any]] = []
	with path.open("r", encoding="utf-8") as f:
		for line_no, raw in enumerate(f, start=1):
			line = raw.strip()
			if not line:
				continue
			try:
				obj = json.loads(line)
			except json.JSONDecodeError as exc:
				raise ValueError(f"Invalid JSON at {path}:{line_no}: {exc}") from exc
			if not isinstance(obj, dict):
				raise ValueError(f"Each JSONL row must be object, got {type(obj)} at {path}:{line_no}")
			rows.append(obj)
	return rows


def append_jsonl(path: Path, obj: dict[str, Any]) -> None:
	"""Append one JSON object as a JSONL line."""
	with path.open("a", encoding="utf-8") as f:
		f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def dump_json(path: Path, obj: dict[str, Any]) -> None:
	"""Write pretty JSON object."""
	with path.open("w", encoding="utf-8") as f:
		json.dump(obj, f, ensure_ascii=False, indent=2)


def extract_code_blocks(text: str) -> list[dict[str, str]]:
	"""Extract fenced code blocks from markdown-like text.

	Returns a list of dicts: {"lang": <language>, "code": <code>}.
	"""
	pattern = re.compile(r"```([a-zA-Z0-9_+-]*)\s*\n(.*?)```", re.DOTALL)
	blocks: list[dict[str, str]] = []
	for match in pattern.finditer(text):
		lang = (match.group(1) or "").strip().lower()
		code = match.group(2).strip()
		if code:
			blocks.append({"lang": lang, "code": code})
	return blocks


def pick_python_code(blocks: list[dict[str, str]]) -> str | None:
	"""Select the best candidate python code block."""
	if not blocks:
		return None

	preferred_langs = {"python", "py"}
	preferred = [b["code"] for b in blocks if b.get("lang") in preferred_langs]
	if preferred:
		return max(preferred, key=len)

	# Fall back to unlabeled code block if no explicit Python block exists.
	unlabeled = [b["code"] for b in blocks if not b.get("lang")]
	if unlabeled:
		return max(unlabeled, key=len)

	return max((b["code"] for b in blocks), key=len)


def extract_numeric_candidates(text: str) -> list[float]:
	"""Extract numeric candidates from text for lightweight answer checks."""
	cleaned = text.replace(",", "")
	nums = re.findall(r"(?<![A-Za-z0-9_])[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", cleaned)
	values: list[float] = []
	for num in nums:
		try:
			values.append(float(num))
		except ValueError:
			continue
	return values


def _strip_ansi(text: str) -> str:
	"""Remove ANSI escape sequences from command output."""
	return re.sub(r"\x1B\[[0-?]*[ -/]*[@-~]", "", text)


def extract_best_numeric_answer(text: str) -> dict[str, Any]:
	"""Extract a numeric answer with simple confidence heuristics.

	Heuristic priority:
	1) Last number on the last keyword line (answer/final/objective/result/optimal).
	2) Last number on the last line that has exactly one numeric token.
	3) Fallback to last numeric token in the whole text.
	"""
	cleaned = _strip_ansi(text)
	lines = [ln.strip() for ln in cleaned.splitlines() if ln.strip()]
	keyword_re = re.compile(r"\b(answer|final|objective|optimal|result|obj\.?|minimum|maximum|min|max)\b", re.IGNORECASE)

	for line in reversed(lines):
		if not keyword_re.search(line):
			continue
		cands = extract_numeric_candidates(line)
		if cands:
			return {
				"value": cands[-1],
				"strategy": "keyword_line_last_number",
				"line": line,
				"line_candidates": cands,
			}

	for line in reversed(lines):
		cands = extract_numeric_candidates(line)
		if len(cands) == 1:
			return {
				"value": cands[0],
				"strategy": "single_number_line",
				"line": line,
				"line_candidates": cands,
			}

	all_cands = extract_numeric_candidates(cleaned)
	if all_cands:
		return {
			"value": all_cands[-1],
			"strategy": "fallback_last_number",
			"line": None,
			"line_candidates": None,
		}

	return {
		"value": None,
		"strategy": "no_number_found",
		"line": None,
		"line_candidates": None,
	}


def run_python_code_subprocess(code: str, work_dir: Path, timeout_s: int = 120) -> dict[str, Any]:
	"""Run generated Python code via subprocess with current Python interpreter.

	Using ``sys.executable`` ensures we run under the active conda environment.
	"""
	ensure_dir(work_dir)
	start = time.time()

	with tempfile.NamedTemporaryFile(
		mode="w",
		suffix=".py",
		prefix="generated_",
		dir=work_dir,
		encoding="utf-8",
		delete=False,
	) as tmp:
		tmp.write(code)
		script_path = Path(tmp.name)

	cmd = [sys.executable, str(script_path)]
	try:
		proc = subprocess.run(
			cmd,
			cwd=str(work_dir),
			stdout=subprocess.PIPE,
			stderr=subprocess.PIPE,
			text=True,
			timeout=timeout_s,
			check=False,
		)
		success = proc.returncode == 0
		return {
			"success": success,
			"returncode": proc.returncode,
			"stdout": proc.stdout,
			"stderr": proc.stderr,
			"duration_s": round(time.time() - start, 4),
			"command": cmd,
			"script_path": str(script_path),
			"timeout": False,
		}
	except subprocess.TimeoutExpired as exc:
		return {
			"success": False,
			"returncode": None,
			"stdout": exc.stdout or "",
			"stderr": exc.stderr or "",
			"duration_s": round(time.time() - start, 4),
			"command": cmd,
			"script_path": str(script_path),
			"timeout": True,
			"error": f"Execution timed out after {timeout_s}s",
		}
