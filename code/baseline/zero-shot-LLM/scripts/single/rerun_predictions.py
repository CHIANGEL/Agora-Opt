"""Rerun extracted code for an existing run and rebuild prediction-related artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from common import (
	dump_json,
	extract_best_numeric_answer,
	read_jsonl,
	run_python_code_subprocess,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BENCH = ROOT / "benchmarks" / "IndustryOR_clean.jsonl"


def compare_answer(pred: float | None, gold: Any, rtol: float = 1e-3) -> dict[str, Any]:
	try:
		gold_float = float(gold)
	except (TypeError, ValueError):
		gold_float = None

	if pred is None or gold_float is None:
		return {
			"gold_answer": gold,
			"pred_answer": pred,
			"is_match": False,
			"rel_error": None,
			"abs_error": None,
		}

	abs_error = abs(pred - gold_float)
	denom = abs(gold_float)
	if denom == 0.0:
		rel_error = 0.0 if abs_error == 0.0 else float("inf")
	else:
		rel_error = abs_error / denom
	return {
		"gold_answer": gold_float,
		"pred_answer": pred,
		"is_match": rel_error <= rtol,
		"rel_error": rel_error,
		"abs_error": abs_error,
	}


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


def load_json(path: Path) -> dict[str, Any]:
	return json.loads(path.read_text(encoding="utf-8"))


def build_arg_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description="Rerun extracted code and rebuild prediction artifacts")
	parser.add_argument("--run-dir", type=Path, required=True)
	parser.add_argument("--input-file", type=Path, default=DEFAULT_BENCH)
	parser.add_argument("--exec-timeout-s", type=int, default=300)
	parser.add_argument("--answer-rtol", type=float, default=1e-3)
	return parser


def main() -> int:
	args = build_arg_parser().parse_args()
	run_dir = args.run_dir
	if not run_dir.exists():
		raise FileNotFoundError(f"Run directory not found: {run_dir}")

	questions = read_jsonl(args.input_file)
	questions_by_id = {int(q.get("id", -1)): q for q in questions if int(q.get("id", -1)) >= 0}
	rows: list[dict[str, Any]] = []

	for idx, qid in enumerate(sorted(questions_by_id), start=1):
		question = questions_by_id[qid]
		qdir = run_dir / f"q_{qid:04d}"
		detail_path = qdir / "detail.json"
		code_path = qdir / "extracted_code.py"
		response_path = qdir / "response.txt"
		stdout_path = qdir / "execution_stdout.txt"
		stderr_path = qdir / "execution_stderr.txt"
		optimal_solution_path = qdir / "optimal_solution.txt"
		optimal_value_path = qdir / "optimal_value.txt"

		detail: dict[str, Any] = {}
		if detail_path.exists():
			detail = load_json(detail_path)

		response_text = ""
		if response_path.exists():
			response_text = response_path.read_text(encoding="utf-8")
		elif isinstance(detail.get("request"), dict):
			response_text = str(detail.get("request", {}).get("response_text", ""))

		request_success = bool(response_text.strip()) or bool(detail.get("request", {}).get("success"))
		exec_result: dict[str, Any] | None = None
		pred: float | None = None
		pred_source: str | None = None
		pred_strategy: str | None = None
		selected_code = None

		if code_path.exists():
			selected_code = code_path.read_text(encoding="utf-8")
			print(f"[{idx}/{len(questions_by_id)}] rerun qid={qid}", flush=True)
			exec_result = run_python_code_subprocess(
				code=selected_code,
				work_dir=qdir,
				timeout_s=args.exec_timeout_s,
			)
			stdout = str(exec_result.get("stdout", ""))
			parsed = extract_best_numeric_answer(stdout)
			if parsed.get("value") is not None:
				pred = float(parsed["value"])
				pred_source = "execution_stdout"
				pred_strategy = str(parsed.get("strategy"))
		else:
			print(f"[{idx}/{len(questions_by_id)}] skip qid={qid} (missing extracted_code.py)", flush=True)

		if pred is None and response_text:
			parsed = extract_best_numeric_answer(response_text)
			if parsed.get("value") is not None:
				pred = float(parsed["value"])
				pred_source = "response_text"
				pred_strategy = str(parsed.get("strategy"))

		compare = compare_answer(pred, question.get("answer"), rtol=args.answer_rtol)
		code_extracted = bool(selected_code and str(selected_code).strip())

		detail["question"] = question
		detail["request"] = {
			"success": request_success,
			"error": detail.get("request", {}).get("error") if isinstance(detail.get("request"), dict) else None,
			"response_text": response_text,
		}
		detail["code_extraction"] = {
			"block_count": detail.get("code_extraction", {}).get("block_count", 1 if code_extracted else 0),
			"selected_python_code": selected_code,
		}
		detail["execution"] = exec_result
		detail["prediction"] = {
			"value": pred,
			"source": pred_source,
			"parse_strategy": pred_strategy,
		}
		detail["comparison"] = compare
		dump_json(detail_path, detail)

		if exec_result is not None:
			stdout_path.write_text(str(exec_result.get("stdout", "")), encoding="utf-8")
			stderr_path.write_text(str(exec_result.get("stderr", "")), encoding="utf-8")

		# Persist current extracted result for quick manual inspection.
		if exec_result is not None:
			optimal_solution_path.write_text(str(exec_result.get("stdout", "")), encoding="utf-8")
		else:
			optimal_solution_path.write_text(response_text, encoding="utf-8")

		if pred is None:
			optimal_value_path.write_text("", encoding="utf-8")
		else:
			optimal_value_path.write_text(str(pred), encoding="utf-8")

		row = {
			"question_id": qid,
			"difficulty": question.get("difficulty"),
			"request_success": request_success,
			"code_extracted": code_extracted,
			"execution_success": bool(exec_result and exec_result.get("success")),
			"prediction": pred,
			"prediction_source": pred_source,
			"gold_answer": question.get("answer"),
			"is_match": compare.get("is_match"),
			"rel_error": compare.get("rel_error"),
			"abs_error": compare.get("abs_error"),
			"detail_file": str((qdir / "detail.json").relative_to(run_dir)),
			"response_file": str((qdir / "response.txt").relative_to(run_dir)) if response_path.exists() else None,
			"code_file": str((qdir / "extracted_code.py").relative_to(run_dir)) if code_path.exists() else None,
			"execution_stdout_file": str((qdir / "execution_stdout.txt").relative_to(run_dir)) if exec_result else None,
			"execution_stderr_file": str((qdir / "execution_stderr.txt").relative_to(run_dir)) if exec_result else None,
		}
		rows.append(row)

	summary_path = run_dir / "summary.jsonl"
	with summary_path.open("w", encoding="utf-8") as f:
		for row in rows:
			f.write(json.dumps(row, ensure_ascii=False) + "\n")

	run_report = {
		"run_dir": str(run_dir),
		"model": (load_json(run_dir / "run_report.json").get("model") if (run_dir / "run_report.json").exists() else None),
		"num_questions": len(rows),
		"request_success_count": sum(1 for r in rows if r.get("request_success")),
		"execution_success_count": sum(1 for r in rows if r.get("execution_success")),
		"match_count": sum(1 for r in rows if r.get("is_match")),
		"summary_file": str(summary_path),
	}
	dump_json(run_dir / "run_report.json", run_report)

	eval_report = summarize_rows(rows, answer_rtol=args.answer_rtol)
	eval_report["run_dir"] = str(run_dir)
	eval_report["summary_file"] = str(summary_path)
	dump_json(run_dir / "eval_report.json", eval_report)

	print(json.dumps({
		"run_dir": str(run_dir),
		"num_questions": len(rows),
		"execution_success_count": run_report["execution_success_count"],
		"match_count": run_report["match_count"],
	}, ensure_ascii=False, indent=2))
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
