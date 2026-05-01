"""Rebuild eval_report.json and run_report.json from summary.jsonl only.

This script does not call model APIs and does not run code execution.
It only computes statistics from an existing summary file.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from common import dump_json, read_jsonl


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_HISTORY = ROOT / "history"


def _ratio(x: int, y: int) -> float:
	return round(x / y, 4) if y else 0.0


def summarize_eval(rows: list[dict[str, Any]], answer_rtol: float) -> dict[str, Any]:
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

	return {
		"num_questions": n,
		"answer_rtol": answer_rtol,
		"request_success": {"count": request_success, "ratio": _ratio(request_success, n)},
		"code_extracted": {"count": code_extracted, "ratio": _ratio(code_extracted, n)},
		"execution_success": {"count": execution_success, "ratio": _ratio(execution_success, n)},
		"answer_match": {"count": is_match, "ratio": _ratio(is_match, n)},
		"by_difficulty": by_difficulty,
		"bad_cases": bad_cases,
	}


def summarize_run(rows: list[dict[str, Any]], run_dir: Path, model: str, summary_file: Path) -> dict[str, Any]:
	model_precheck_path = run_dir / "model_precheck.json"
	model_precheck: dict[str, Any] | None = None
	if model_precheck_path.exists():
		try:
			model_precheck = json.loads(model_precheck_path.read_text(encoding="utf-8"))
		except Exception:  # noqa: BLE001
			model_precheck = None

	return {
		"run_dir": str(run_dir),
		"model": model,
		"model_precheck": model_precheck,
		"num_questions": len(rows),
		"request_success_count": sum(1 for r in rows if bool(r.get("request_success"))),
		"execution_success_count": sum(1 for r in rows if bool(r.get("execution_success"))),
		"match_count": sum(1 for r in rows if bool(r.get("is_match"))),
		"summary_file": str(summary_file),
	}


def infer_model(run_dir: Path, model_arg: str | None) -> str:
	if model_arg and model_arg.strip():
		return model_arg.strip()
	# history/<model>/<run_name>
	if run_dir.parent.name:
		return run_dir.parent.name
	return "unknown_model"


def build_arg_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description="Rebuild eval_report.json and run_report.json from summary.jsonl")
	parser.add_argument("--run-dir", type=Path, required=True, help="Run directory containing summary.jsonl")
	parser.add_argument("--summary-file", type=Path, default=None, help="Override summary file path")
	parser.add_argument("--model", type=str, default=None, help="Override model name in run_report")
	parser.add_argument("--answer-rtol", type=float, default=1e-3)
	parser.add_argument("--dry-run", action="store_true", help="Print reports only, do not write files")
	return parser


def main() -> int:
	args = build_arg_parser().parse_args()
	run_dir = args.run_dir
	if not run_dir.exists():
		raise FileNotFoundError(f"Run directory not found: {run_dir}")

	summary_file = args.summary_file if args.summary_file is not None else run_dir / "summary.jsonl"
	if not summary_file.exists():
		raise FileNotFoundError(f"summary.jsonl not found: {summary_file}")

	rows = read_jsonl(summary_file)
	model = infer_model(run_dir, args.model)

	eval_report = summarize_eval(rows, answer_rtol=args.answer_rtol)
	eval_report["run_dir"] = str(run_dir)
	eval_report["summary_file"] = str(summary_file)

	run_report = summarize_run(rows, run_dir=run_dir, model=model, summary_file=summary_file)

	if not args.dry_run:
		dump_json(run_dir / "eval_report.json", eval_report)
		dump_json(run_dir / "run_report.json", run_report)

	print("[rebuild-reports] done")
	print(f"[rebuild-reports] run_dir={run_dir}")
	print(f"[rebuild-reports] summary_file={summary_file}")
	print(f"[rebuild-reports] model={model}")
	print(json.dumps({"run_report": run_report, "eval_report": eval_report}, ensure_ascii=False, indent=2))
	if args.dry_run:
		print("[rebuild-reports] dry_run=True, no files written")
	else:
		print(f"[rebuild-reports] wrote={run_dir / 'run_report.json'}")
		print(f"[rebuild-reports] wrote={run_dir / 'eval_report.json'}")

	return 0


if __name__ == "__main__":
	raise SystemExit(main())
