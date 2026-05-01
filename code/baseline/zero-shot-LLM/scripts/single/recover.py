"""Recover run directory by syncing summary with files and rerunning missing items.

Usage:
- Input: an existing run directory under history.
- Step 1: sync request_success/code_extracted based on response.txt/extracted_code.py.
- Step 2: rerun only question IDs missing these files.
- Output: rewrite summary.jsonl in-place and emit recover_report.json.
"""

from __future__ import annotations

import argparse
import json
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from common import read_jsonl
from generate import process_one_question


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BENCH = ROOT / "benchmarks" / "IndustryOR_clean.jsonl"


def _question_sort_key(row: dict[str, Any]) -> int:
	try:
		return int(row.get("question_id", -1))
	except Exception:  # noqa: BLE001
		return -1


def _load_questions_by_id(input_file: Path) -> dict[int, dict[str, Any]]:
	rows = read_jsonl(input_file)
	out: dict[int, dict[str, Any]] = {}
	for row in rows:
		qid = int(row.get("id", -1))
		if qid >= 0:
			out[qid] = row
	return out


def _find_run_meta(run_dir: Path) -> dict[str, Any]:
	# Try to recover run settings from any detail file generated in this run.
	for detail_path in sorted(run_dir.glob("q_*/detail.json")):
		try:
			detail = json.loads(detail_path.read_text(encoding="utf-8"))
			run_meta = detail.get("run_meta")
			if isinstance(run_meta, dict):
				return run_meta
		except Exception:  # noqa: BLE001
			continue
	return {}


def _rewrite_summary(summary_path: Path, rows: list[dict[str, Any]]) -> None:
	with summary_path.open("w", encoding="utf-8") as f:
		for row in rows:
			f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _count_missing_files(rows: list[dict[str, Any]]) -> int:
	return sum(1 for row in rows if (not bool(row.get("request_success"))) or (not bool(row.get("code_extracted"))))


def _sync_summary_by_files(
	run_dir: Path,
	summary_rows: list[dict[str, Any]],
	questions_by_id: dict[int, dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[int]]:
	"""Rewrite request_success/code_extracted by actual files under q_xxxx dirs.

	Also ensures every question id from benchmark has one row in summary.
	Returns synced rows and ids missing response/code files.
	"""
	by_qid: dict[int, dict[str, Any]] = {}
	for row in summary_rows:
		qid = int(row.get("question_id", -1))
		if qid >= 0:
			by_qid[qid] = row

	missing_ids: list[int] = []
	synced_rows: list[dict[str, Any]] = []

	for qid in sorted(questions_by_id):
		question = questions_by_id[qid]
		qdir = run_dir / f"q_{qid:04d}"
		response_exists = (qdir / "response.txt").exists()
		code_exists = (qdir / "extracted_code.py").exists()

		row = by_qid.get(qid, {
			"question_id": qid,
			"difficulty": question.get("difficulty"),
			"execution_success": False,
			"prediction": None,
			"prediction_source": None,
			"gold_answer": question.get("answer"),
			"is_match": False,
			"rel_error": None,
			"abs_error": None,
			"detail_file": f"q_{qid:04d}/detail.json",
			"execution_stdout_file": None,
			"execution_stderr_file": None,
		})

		row["difficulty"] = question.get("difficulty")
		row["gold_answer"] = question.get("answer")
		row["request_success"] = bool(response_exists)
		row["code_extracted"] = bool(code_exists)
		row["response_file"] = f"q_{qid:04d}/response.txt" if response_exists else None
		row["code_file"] = f"q_{qid:04d}/extracted_code.py" if code_exists else None

		if not response_exists or not code_exists:
			missing_ids.append(qid)

		synced_rows.append(row)

	synced_rows.sort(key=_question_sort_key)
	return synced_rows, missing_ids


def build_arg_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description="Recover failed items for one run directory")
	parser.add_argument("--run-dir", type=Path, required=True, help="Run directory that contains summary.jsonl")
	parser.add_argument("--input-file", type=Path, default=DEFAULT_BENCH, help="Benchmark JSONL input file")
	parser.add_argument("--max-rounds", type=int, default=3, help="Maximum recovery rounds")
	parser.add_argument("--temperature", type=float, default=None)
	parser.add_argument("--max-tokens", type=int, default=None)
	parser.add_argument("--maximum-retries", type=int, default=None)
	parser.add_argument("--timeout-s", type=float, default=None)
	parser.add_argument("--exec-timeout-s", type=int, default=None)
	parser.add_argument("--answer-rtol", type=float, default=None)
	parser.add_argument("--model", type=str, default=None, help="Override model if needed")
	parser.add_argument(
		"--only-sync",
		action="store_true",
		help="Only sync summary request_success/code_extracted by files; do not rerun missing ids",
	)
	return parser


def main() -> int:
	args = build_arg_parser().parse_args()
	run_dir = args.run_dir
	summary_path = run_dir / "summary.jsonl"

	if not run_dir.exists():
		raise FileNotFoundError(f"Run directory not found: {run_dir}")
	if not summary_path.exists():
		raise FileNotFoundError(f"Missing summary.jsonl in run directory: {summary_path}")

	questions_by_id = _load_questions_by_id(args.input_file)
	summary_rows = read_jsonl(summary_path)

	run_meta = _find_run_meta(run_dir)
	model = args.model or str(run_meta.get("model") or "")
	if not model:
		raise RuntimeError("Cannot determine model. Please pass --model explicitly")

	ns = SimpleNamespace(
		model=model,
		temperature=(args.temperature if args.temperature is not None else float(run_meta.get("temperature", 0.01))),
		max_tokens=(args.max_tokens if args.max_tokens is not None else int(run_meta.get("max_tokens", 4096))),
		maximum_retries=(
			args.maximum_retries if args.maximum_retries is not None else int(run_meta.get("maximum_retries", 3))
		),
		timeout_s=(args.timeout_s if args.timeout_s is not None else run_meta.get("timeout_s")),
		exec_timeout_s=(
			args.exec_timeout_s if args.exec_timeout_s is not None else int(run_meta.get("exec_timeout_s", 300))
		),
		answer_rtol=(args.answer_rtol if args.answer_rtol is not None else float(run_meta.get("answer_rtol", 1e-3))),
	)

	summary_rows, missing_ids = _sync_summary_by_files(run_dir, summary_rows, questions_by_id)
	_rewrite_summary(summary_path, summary_rows)

	before_failed = _count_missing_files(summary_rows)
	print(f"[recover] run_dir={run_dir}")
	print(f"[recover] missing_files_before={before_failed}")

	total_recovered = 0
	lock = threading.Lock()
	temp_summary_path = run_dir / "_recover_append.jsonl"

	if args.only_sync:
		report = {
			"run_dir": str(run_dir),
			"model": model,
			"missing_files_before": before_failed,
			"missing_files_after": before_failed,
			"recovered_count": 0,
			"max_rounds": int(args.max_rounds),
			"only_sync": True,
			"missing_ids": missing_ids,
			"used_args": {
				"temperature": ns.temperature,
				"max_tokens": ns.max_tokens,
				"maximum_retries": ns.maximum_retries,
				"timeout_s": ns.timeout_s,
				"exec_timeout_s": ns.exec_timeout_s,
				"answer_rtol": ns.answer_rtol,
			},
		}
		(run_dir / "recover_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
		print(json.dumps(report, ensure_ascii=False, indent=2))
		return 0

	for round_idx in range(1, max(1, args.max_rounds) + 1):
		summary_rows, missing_ids = _sync_summary_by_files(run_dir, summary_rows, questions_by_id)
		_rewrite_summary(summary_path, summary_rows)
		if not missing_ids:
			break

		print(f"[recover] round={round_idx} pending_missing_files={len(missing_ids)}")
		if temp_summary_path.exists():
			temp_summary_path.unlink()

		updates: dict[int, dict[str, Any]] = {}
		for qid in missing_ids:
			question = questions_by_id.get(qid)
			if question is None:
				print(f"[recover] skip id={qid}: question not found in input file")
				continue
			row = process_one_question(
				question=question,
				args=ns,
				run_dir=run_dir,
				summary_path=temp_summary_path,
				summary_lock=lock,
			)
			updates[qid] = row

		# Merge latest rows into summary and rewrite in-place.
		merged: dict[int, dict[str, Any]] = {}
		for row in summary_rows:
			qid = int(row.get("question_id", -1))
			if qid >= 0:
				merged[qid] = row
		for qid, row in updates.items():
			prev = merged.get(qid, {})
			prev_missing = (not bool(prev.get("request_success"))) or (not bool(prev.get("code_extracted")))
			new_ok = bool(row.get("request_success")) and bool(row.get("code_extracted"))
			if prev_missing and new_ok:
				total_recovered += 1
			merged[qid] = row

		summary_rows = [merged[qid] for qid in sorted(merged)]
		_rewrite_summary(summary_path, summary_rows)

	summary_rows, missing_ids = _sync_summary_by_files(run_dir, summary_rows, questions_by_id)
	_rewrite_summary(summary_path, summary_rows)
	after_failed = _count_missing_files(summary_rows)
	report = {
		"run_dir": str(run_dir),
		"model": model,
		"missing_files_before": before_failed,
		"missing_files_after": after_failed,
		"recovered_count": total_recovered,
		"max_rounds": int(args.max_rounds),
		"missing_ids_after": missing_ids,
		"used_args": {
			"temperature": ns.temperature,
			"max_tokens": ns.max_tokens,
			"maximum_retries": ns.maximum_retries,
			"timeout_s": ns.timeout_s,
			"exec_timeout_s": ns.exec_timeout_s,
			"answer_rtol": ns.answer_rtol,
		},
	}
	(run_dir / "recover_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

	print(json.dumps(report, ensure_ascii=False, indent=2))
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
