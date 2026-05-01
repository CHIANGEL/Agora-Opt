"""Evaluate saved run summaries under history directory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from common import dump_json, read_jsonl


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_HISTORY = ROOT / "history"


def resolve_run_dir(history_root: Path, model: str | None, run_dir: Path | None, latest: bool) -> Path:
	if run_dir is not None:
		if not run_dir.exists():
			raise FileNotFoundError(f"Run directory not found: {run_dir}")
		return run_dir

	if not model:
		raise ValueError("Provide --run-dir or (--model with optional --latest)")

	model_dir = history_root / model
	if not model_dir.exists():
		# Try safe folder fallback if model includes non-path chars.
		alternatives = [p for p in history_root.iterdir() if p.is_dir() and p.name == model]
		if not alternatives:
			raise FileNotFoundError(f"Model directory not found under history: {model_dir}")

	runs = sorted([p for p in model_dir.iterdir() if p.is_dir()])
	if not runs:
		raise FileNotFoundError(f"No runs found under: {model_dir}")

	if latest:
		return runs[-1]

	raise ValueError("When --run-dir is omitted, currently only --latest mode is supported")


def summarize_rows(rows: list[dict[str, Any]], answer_rtol: float) -> dict[str, Any]:
	n = len(rows)
	request_success = sum(1 for r in rows if bool(r.get("request_success")))
	code_extracted = sum(1 for r in rows if bool(r.get("code_extracted")))
	execution_success = sum(1 for r in rows if bool(r.get("execution_success")))
	is_match = sum(1 for r in rows if bool(r.get("is_match")))

	by_difficulty: dict[str, dict[str, int]] = {}
	for r in rows:
		diff = str(r.get("difficulty", "Unknown"))
		if diff not in by_difficulty:
			by_difficulty[diff] = {"total": 0, "match": 0, "exec_success": 0}
		by_difficulty[diff]["total"] += 1
		if bool(r.get("is_match")):
			by_difficulty[diff]["match"] += 1
		if bool(r.get("execution_success")):
			by_difficulty[diff]["exec_success"] += 1

	bad_cases = [
		{
			"question_id": r.get("question_id"),
			"difficulty": r.get("difficulty"),
			"request_success": r.get("request_success"),
			"code_extracted": r.get("code_extracted"),
			"execution_success": r.get("execution_success"),
			"prediction": r.get("prediction"),
			"gold_answer": r.get("gold_answer"),
			"abs_error": r.get("abs_error"),
			"detail_file": r.get("detail_file"),
		}
		for r in rows
		if not bool(r.get("is_match"))
	]

	def ratio(x: int, y: int) -> float:
		return round(x / y, 4) if y else 0.0

	return {
		"num_questions": n,
		"answer_rtol": answer_rtol,
		"request_success": {"count": request_success, "ratio": ratio(request_success, n)},
		"code_extracted": {"count": code_extracted, "ratio": ratio(code_extracted, n)},
		"execution_success": {"count": execution_success, "ratio": ratio(execution_success, n)},
		"answer_match": {"count": is_match, "ratio": ratio(is_match, n)},
		"by_difficulty": by_difficulty,
		"bad_cases": bad_cases,
	}


def build_arg_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description="Evaluate a single history run.")
	parser.add_argument("--history-root", type=Path, default=DEFAULT_HISTORY)
	parser.add_argument("--model", type=str, default=None, help="Model folder name under history")
	parser.add_argument("--run-dir", type=Path, default=None, help="Exact run directory path")
	parser.add_argument("--latest", action="store_true", help="Use latest run under history/<model>")
	parser.add_argument("--summary-file", type=Path, default=None, help="Override summary file path")
	parser.add_argument("--answer-rtol", type=float, default=1e-3)
	return parser


def main() -> None:
	args = build_arg_parser().parse_args()
	run_dir = resolve_run_dir(args.history_root, args.model, args.run_dir, args.latest)

	summary_file = args.summary_file if args.summary_file else run_dir / "summary.jsonl"
	if not summary_file.exists():
		raise FileNotFoundError(f"summary.jsonl not found: {summary_file}")

	rows = read_jsonl(summary_file)
	report = summarize_rows(rows, answer_rtol=args.answer_rtol)
	report["run_dir"] = str(run_dir)
	report["summary_file"] = str(summary_file)

	output_file = run_dir / "eval_report.json"
	dump_json(output_file, report)

	print(json.dumps(report, ensure_ascii=False, indent=2))
	print(f"\nSaved evaluation report: {output_file}")


if __name__ == "__main__":
	main()
