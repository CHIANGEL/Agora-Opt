"""Call LLM API on benchmark items and persist structured results."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import threading
from pathlib import Path
from typing import Any

from api_utils import check_model_available, get_response
from common import (
	append_jsonl,
	build_optimization_prompt,
	dump_json,
	ensure_dir,
	extract_code_blocks,
	extract_best_numeric_answer,
	now_ts_compact,
	pick_python_code,
	read_jsonl,
	run_python_code_subprocess,
	safe_model_name,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BENCH = ROOT / "benchmarks" / "IndustryOR_clean.jsonl"
DEFAULT_HISTORY = ROOT / "history"


def parse_direct_input(input_json: str) -> list[dict[str, Any]]:
	obj = json.loads(input_json)
	if isinstance(obj, dict):
		return [obj]
	if isinstance(obj, list):
		rows = [x for x in obj if isinstance(x, dict)]
		if len(rows) != len(obj):
			raise ValueError("--input-json must be a JSON object or a list of JSON objects")
		return rows
	raise ValueError("--input-json must be a JSON object or a list of JSON objects")


def filter_questions(rows: list[dict[str, Any]], question_ids: set[int] | None, max_items: int | None) -> list[dict[str, Any]]:
	selected = rows
	if question_ids is not None:
		selected = [r for r in selected if int(r.get("id", -1)) in question_ids]
	if max_items is not None and max_items > 0:
		selected = selected[:max_items]
	return selected


def create_run_dir(output_dir: Path, model: str, experiment_name: str | None) -> Path:
	model_dir = ensure_dir(output_dir / safe_model_name(model))
	run_name = experiment_name.strip() if experiment_name else now_ts_compact()
	run_dir = model_dir / run_name
	if run_dir.exists():
		run_dir = model_dir / f"{run_name}_{now_ts_compact()}"
	ensure_dir(run_dir)
	return run_dir


def parse_answer_value(text: str) -> tuple[float | None, str]:
	parsed = extract_best_numeric_answer(text)
	return parsed.get("value"), str(parsed.get("strategy"))


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


def process_one_question(
	question: dict[str, Any],
	args: argparse.Namespace,
	run_dir: Path,
	summary_path: Path,
	summary_lock: threading.Lock,
) -> dict[str, Any]:
	qid = int(question.get("id", -1))
	qdir = ensure_dir(run_dir / f"q_{qid:04d}")

	prompt = build_optimization_prompt(str(question.get("description", "")))
	request_ok = True
	request_error: str | None = None
	response_text = ""

	try:
		response_text = get_response(
			prompt=prompt,
			model=args.model,
			temperature=args.temperature,
			max_tokens=args.max_tokens,
			maximum_retries=args.maximum_retries,
			timeout_s=args.timeout_s,
		)
	except Exception as exc:  # noqa: BLE001
		request_ok = False
		request_error = str(exc)

	blocks = extract_code_blocks(response_text) if request_ok else []
	picked_code = pick_python_code(blocks)
	code_ok = picked_code is not None
	exec_result: dict[str, Any] | None = None

	if request_ok and code_ok and picked_code is not None:
		exec_result = run_python_code_subprocess(
			code=picked_code,
			work_dir=qdir,
			timeout_s=args.exec_timeout_s,
		)

	# Prefer numeric answer from execution output, fallback to response text.
	pred_answer: float | None = None
	answer_from = None
	answer_parse_strategy: str | None = None
	if exec_result and str(exec_result.get("stdout", "")).strip():
		pred_answer, answer_parse_strategy = parse_answer_value(str(exec_result.get("stdout", "")))
		answer_from = "execution_stdout" if pred_answer is not None else None
	if pred_answer is None and request_ok:
		pred_answer, answer_parse_strategy = parse_answer_value(response_text)
		answer_from = "response_text" if pred_answer is not None else None

	compare = compare_answer(pred_answer, question.get("answer"), rtol=args.answer_rtol)

	detail = {
		"run_meta": {
			"model": args.model,
			"temperature": args.temperature,
			"max_tokens": args.max_tokens,
			"maximum_retries": args.maximum_retries,
			"timeout_s": args.timeout_s,
			"exec_timeout_s": args.exec_timeout_s,
			"answer_rtol": args.answer_rtol,
		},
		"question": question,
		"request": {
			"success": request_ok,
			"error": request_error,
			"response_text": response_text,
		},
		"code_extraction": {
			"block_count": len(blocks),
			"selected_python_code": picked_code,
		},
		"execution": exec_result,
		"prediction": {
			"value": pred_answer,
			"source": answer_from,
			"parse_strategy": answer_parse_strategy,
		},
		"comparison": compare,
	}

	# Per-question files.
	if request_ok:
		(qdir / "response.txt").write_text(response_text, encoding="utf-8")
	if code_ok and picked_code is not None:
		(qdir / "extracted_code.py").write_text(picked_code, encoding="utf-8")
	if exec_result is not None:
		(qdir / "execution_stdout.txt").write_text(str(exec_result.get("stdout", "")), encoding="utf-8")
		(qdir / "execution_stderr.txt").write_text(str(exec_result.get("stderr", "")), encoding="utf-8")

	# Persist current extracted result for quick manual inspection.
	if exec_result is not None:
		(qdir / "optimal_solution.txt").write_text(str(exec_result.get("stdout", "")), encoding="utf-8")
	else:
		(qdir / "optimal_solution.txt").write_text(response_text, encoding="utf-8")

	if pred_answer is None:
		(qdir / "optimal_value.txt").write_text("", encoding="utf-8")
	else:
		(qdir / "optimal_value.txt").write_text(str(pred_answer), encoding="utf-8")
	dump_json(qdir / "detail.json", detail)

	summary_row = {
		"question_id": qid,
		"difficulty": question.get("difficulty"),
		"request_success": request_ok,
		"code_extracted": code_ok,
		"execution_success": bool(exec_result and exec_result.get("success")),
		"prediction": pred_answer,
		"prediction_source": answer_from,
		"gold_answer": question.get("answer"),
		"is_match": compare.get("is_match"),
		"rel_error": compare.get("rel_error"),
		"abs_error": compare.get("abs_error"),
		"detail_file": str((qdir / "detail.json").relative_to(run_dir)),
		"response_file": str((qdir / "response.txt").relative_to(run_dir)) if request_ok else None,
		"code_file": str((qdir / "extracted_code.py").relative_to(run_dir)) if code_ok else None,
		"execution_stdout_file": str((qdir / "execution_stdout.txt").relative_to(run_dir)) if exec_result else None,
		"execution_stderr_file": str((qdir / "execution_stderr.txt").relative_to(run_dir)) if exec_result else None,
	}
	with summary_lock:
		append_jsonl(summary_path, summary_row)
	return summary_row


def split_into_batches(items: list[dict[str, Any]], num_batches: int) -> list[list[dict[str, Any]]]:
	"""Split items into round-robin batches for worker-level sequential processing."""
	if num_batches <= 1:
		return [items]
	batches: list[list[dict[str, Any]]] = [[] for _ in range(num_batches)]
	for idx, item in enumerate(items):
		batches[idx % num_batches].append(item)
	return [batch for batch in batches if batch]


def build_arg_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description="Generate model outputs and persist execution results.")
	parser.add_argument("--model", required=True, help="Model name for API request")
	parser.add_argument("--input-file", type=Path, required=True, help="JSONL benchmark file")
	parser.add_argument("--output-dir", type=Path, required=True, help="Root directory to save run history")
	parser.add_argument("--experiment-name", type=str, required=True, help="Optional run folder name under history/<model>")
	parser.add_argument("--question-ids", type=str, default=None, help="Comma-separated ids, e.g. 1,2,3")
	parser.add_argument("--max-items", type=int, default=None, help="Maximum number of questions to run")
	parser.add_argument("--temperature", type=float, default=0.01)
	parser.add_argument("--max-tokens", type=int, default=4096)
	parser.add_argument("--maximum-retries", type=int, default=10)
	parser.add_argument("--timeout-s", type=float, default=None)
	parser.add_argument("--exec-timeout-s", type=int, default=300)
	parser.add_argument("--answer-rtol", type=float, default=1e-3, help="Relative error threshold for answer match")
	parser.add_argument("--max-workers", type=int, default=1, help="Number of parallel worker batches")
	parser.add_argument("--precheck-model", action="store_true", help="Run models.list() precheck before generation")
	parser.add_argument(
		"--strict-model-check",
		action="store_true",
		help="Abort run when precheck says model is unavailable",
	)
	return parser


def main() -> None:
	args = build_arg_parser().parse_args()

	all_questions = read_jsonl(args.input_file)

	qids: set[int] | None = None
	if args.question_ids:
		qids = {int(x.strip()) for x in args.question_ids.split(",") if x.strip()}

	selected = filter_questions(all_questions, question_ids=qids, max_items=args.max_items)
	if not selected:
		raise ValueError("No questions selected to run. Check --question-ids / --max-items / input source")

	run_dir = create_run_dir(args.output_dir, args.model, args.experiment_name)
	summary_path = run_dir / "summary.jsonl"

	model_precheck: dict[str, Any] | None = None
	if args.precheck_model:
		model_available = check_model_available(args.model)
		model_precheck = {"ok": model_available, "reason": "Not found" if not model_available else "Available"}
		dump_json(run_dir / "model_precheck.json", model_precheck)
		if not model_precheck.get("ok"):
			print(f"[model-precheck] WARNING: {model_precheck.get('reason')}", flush=True)
			if args.strict_model_check:
				raise RuntimeError("Model precheck failed and --strict-model-check is enabled")

	rows: list[dict[str, Any]] = []
// ... existing code ...


def main() -> None:
	args = build_arg_parser().parse_args()

	if args.input_json:
		all_questions = parse_direct_input(args.input_json)
	else:
		all_questions = read_jsonl(args.input_file)

	qids: set[int] | None = None
	if args.question_ids:
		qids = {int(x.strip()) for x in args.question_ids.split(",") if x.strip()}

	selected = filter_questions(all_questions, question_ids=qids, max_items=args.max_items)
	if not selected:
		raise ValueError("No questions selected to run. Check --question-ids / --max-items / input source")

	run_dir = create_run_dir(args.history_root, args.model, args.experiment_name)
	summary_path = run_dir / "summary.jsonl"

	model_precheck: dict[str, Any] | None = None
	if args.precheck_model:
		model_precheck = check_model_available(args.model, timeout_s=args.timeout_s)
		dump_json(run_dir / "model_precheck.json", model_precheck)
		if not model_precheck.get("ok"):
			print(f"[model-precheck] WARNING: {model_precheck.get('reason')}", flush=True)
			if args.strict_model_check:
				raise RuntimeError("Model precheck failed and --strict-model-check is enabled")

	rows: list[dict[str, Any]] = []
	summary_lock = threading.Lock()
	max_workers = max(1, int(args.max_workers))
	batches = split_into_batches(selected, max_workers)

	if len(batches) == 1:
		for idx, q in enumerate(selected, start=1):
			qid = q.get("id", "unknown")
			print(f"[{idx}/{len(selected)}] Running question id={qid}...", flush=True)
			row = process_one_question(
				question=q,
				args=args,
				run_dir=run_dir,
				summary_path=summary_path,
				summary_lock=summary_lock,
			)
			rows.append(row)
	else:
		print(f"Running with {len(batches)} parallel worker batches...", flush=True)

		def run_batch(worker_idx: int, batch: list[dict[str, Any]]) -> list[dict[str, Any]]:
			batch_rows: list[dict[str, Any]] = []
			for local_idx, q in enumerate(batch, start=1):
				qid = q.get("id", "unknown")
				print(
					f"[worker {worker_idx}] ({local_idx}/{len(batch)}) Running question id={qid}...",
					flush=True,
				)
				row = process_one_question(
					question=q,
					args=args,
					run_dir=run_dir,
					summary_path=summary_path,
					summary_lock=summary_lock,
				)
				batch_rows.append(row)
			return batch_rows

		with concurrent.futures.ThreadPoolExecutor(max_workers=len(batches)) as executor:
			futures = [executor.submit(run_batch, idx + 1, batch) for idx, batch in enumerate(batches)]
			for future in concurrent.futures.as_completed(futures):
				rows.extend(future.result())

	run_report = {
		"run_dir": str(run_dir),
		"model": args.model,
		"model_precheck": model_precheck,
		"num_questions": len(rows),
		"request_success_count": sum(1 for r in rows if r.get("request_success")),
		"execution_success_count": sum(1 for r in rows if r.get("execution_success")),
		"match_count": sum(1 for r in rows if r.get("is_match")),
		"summary_file": str(summary_path),
	}
	dump_json(run_dir / "run_report.json", run_report)

	print("\nRun completed.")
	print(json.dumps(run_report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
	main()

