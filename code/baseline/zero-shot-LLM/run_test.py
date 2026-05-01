"""Run combined benchmark for one or many models with robust recovery.

Flow per model:
1) generate
2) eval
3) recover
4) eval again

The prompt and extraction/execution details remain exactly in scripts/single/generate.py,
so this runner only orchestrates the existing pipeline.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
GENERATE_SCRIPT = SCRIPT_DIR / "scripts" / "single" / "generate.py"
EVAL_SCRIPT = SCRIPT_DIR / "scripts" / "single" / "eval.py"
RECOVER_SCRIPT = SCRIPT_DIR / "scripts" / "single" / "recover.py"

DEFAULT_MAX_WORKERS = 5
DEFAULT_TIMEOUT_S = 500.0
DEFAULT_MAXIMUM_RETRIES = 5
DEFAULT_ANSWER_RTOL = 5e-2
DEFAULT_MAX_TOKENS = 40000


@dataclass
class ModelRunResult:
	model: str
	run_dir: Path | None
	run_report: dict[str, Any] | None
	initial_eval_report: dict[str, Any] | None
	recover_report: dict[str, Any] | None
	eval_report: dict[str, Any] | None
	failed_before_recover: int | None
	failed_after_recover: int | None
	success: bool
	error: str | None


def safe_model_name(model: str) -> str:
	return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in model.strip()) or "unknown_model"


def run_cmd(cmd: list[str | Path], cwd: Path) -> subprocess.CompletedProcess[str]:
	print("[run]", " ".join(map(str, cmd)), flush=True)
	return subprocess.run(
		[str(c) for c in cmd],
		cwd=str(cwd),
		text=True,
		stdout=subprocess.PIPE,
		stderr=subprocess.PIPE,
		check=False,
	)


def load_json(path: Path) -> dict[str, Any]:
	return json.loads(path.read_text(encoding="utf-8"))


def find_latest_run_dir(model: str, experiment_name: str, output_dir: Path) -> Path:
	model_dir = output_dir / safe_model_name(model)
	if not model_dir.exists():
		raise FileNotFoundError(f"Model history directory not found: {model_dir}")

	candidates = [p for p in model_dir.iterdir() if p.is_dir() and p.name.startswith(experiment_name)]
	if not candidates:
		raise FileNotFoundError(f"No run directories found in {model_dir} with prefix {experiment_name}")

	return sorted(candidates, key=lambda p: p.stat().st_mtime)[-1]


def resolve_existing_run_dir(model: str, experiment_name: str, output_dir: Path) -> Path:
	"""Resolve existing run dir for repair flow.

	Priority:
	1) <output_dir>/<safe_model>/<experiment_name>
	2) latest run under <output_dir>/<safe_model>/ with prefix <experiment_name>
	"""
	model_dir = output_dir / safe_model_name(model)
	direct_path = model_dir / experiment_name
	if direct_path.exists() and direct_path.is_dir():
		return direct_path
	return find_latest_run_dir(model, experiment_name, output_dir)


def count_missing_in_summary(summary_path: Path) -> int:
	rows: list[dict[str, Any]] = []
	with summary_path.open("r", encoding="utf-8") as f:
		for line in f:
			line = line.strip()
			if line:
				rows.append(json.loads(line))
	return sum(
		1
		for row in rows
		if not (bool(row.get("request_success")) and bool(row.get("code_extracted")))
	)


def run_eval(run_dir: Path, answer_rtol: float) -> subprocess.CompletedProcess[str]:
	eval_cmd = [
		sys.executable,
		EVAL_SCRIPT,
		"--run-dir",
		run_dir,
		"--answer-rtol",
		answer_rtol,
	]
	return run_cmd(eval_cmd, SCRIPT_DIR)


def run_recover(
	run_dir: Path,
	bench_file: Path,
	max_tokens: int,
	maximum_retries: int,
	timeout_s: float,
) -> subprocess.CompletedProcess[str]:
	recover_cmd = [
		sys.executable,
		RECOVER_SCRIPT,
		"--run-dir",
		run_dir,
		"--input-file",
		bench_file,
		"--max-rounds",
		"3",
		"--max-tokens",
		max_tokens,
		"--maximum-retries",
		maximum_retries,
		"--timeout-s",
		timeout_s,
	]
	return run_cmd(recover_cmd, SCRIPT_DIR)


def run_one_model(model: str, args: argparse.Namespace) -> ModelRunResult:
	"""Run the full generate/eval/recover/eval pipeline for a single model."""
	try:
		generate_cmd = [
			sys.executable,
			GENERATE_SCRIPT,
			"--model",
			model,
			"--input-file",
			args.dataset,
			"--experiment-name",
			Path(args.dataset).stem,
			"--max-workers",
			str(args.max_workers),
			"--timeout-s",
			str(args.timeout_s),
			"--maximum-retries",
			str(args.maximum_retries),
			"--answer-rtol",
			str(args.answer_rtol),
			"--max-tokens",
			str(args.max_tokens),
			"--precheck-model",
            "--output-dir",
            str(args.output_dir),
		]
		gen_proc = run_cmd(generate_cmd, SCRIPT_DIR)
		if gen_proc.returncode != 0:
			return ModelRunResult(
				model=model,
				run_dir=None,
				run_report=None,
				initial_eval_report=None,
				recover_report=None,
				eval_report=None,
				failed_before_recover=None,
				failed_after_recover=None,
				success=False,
				error=f"generate failed (code={gen_proc.returncode}): {gen_proc.stderr.strip()}",
			)

		run_dir = find_latest_run_dir(model, Path(args.dataset).stem, args.output_dir)
		run_report_path = run_dir / "run_report.json"
		if not run_report_path.exists():
			return ModelRunResult(
				model=model,
				run_dir=run_dir,
				run_report=None,
				initial_eval_report=None,
				recover_report=None,
				eval_report=None,
				failed_before_recover=None,
				failed_after_recover=None,
				success=False,
				error=f"Missing run_report.json in {run_dir}",
			)

		summary_path = run_dir / "summary.jsonl"
		failed_before = count_missing_in_summary(summary_path)
		print(f"[first-pass] model={model} missing_files_count={failed_before}", flush=True)

		eval_proc = run_eval(run_dir, args.answer_rtol)
		if eval_proc.returncode != 0:
			return ModelRunResult(
				model=model,
				run_dir=run_dir,
				run_report=load_json(run_report_path),
				initial_eval_report=None,
				recover_report=None,
				eval_report=None,
				failed_before_recover=failed_before,
				failed_after_recover=None,
				success=False,
				error=f"eval failed (code={eval_proc.returncode}): {eval_proc.stderr.strip()}",
			)

		eval_report_path = run_dir / "eval_report.json"
		if not eval_report_path.exists():
			return ModelRunResult(
				model=model,
				run_dir=run_dir,
				run_report=load_json(run_report_path),
				initial_eval_report=None,
				recover_report=None,
				eval_report=None,
				failed_before_recover=failed_before,
				failed_after_recover=None,
				success=False,
				error=f"Missing eval_report.json in {run_dir}",
			)

		initial_eval = load_json(eval_report_path)

		recover_proc = run_recover(
			run_dir=run_dir,
			bench_file=args.dataset,
			max_tokens=args.max_tokens,
			maximum_retries=args.maximum_retries,
			timeout_s=args.timeout_s,
		)
		if recover_proc.returncode != 0:
			return ModelRunResult(
				model=model,
				run_dir=run_dir,
				run_report=load_json(run_report_path),
				initial_eval_report=initial_eval,
				recover_report=None,
				eval_report=None,
				failed_before_recover=failed_before,
				failed_after_recover=None,
				success=False,
				error=f"recover failed (code={recover_proc.returncode}): {recover_proc.stderr.strip()}",
			)

		recover_report_path = run_dir / "recover_report.json"
		recover_report = load_json(recover_report_path) if recover_report_path.exists() else None

		eval_proc_2 = run_eval(run_dir, args.answer_rtol)
		if eval_proc_2.returncode != 0:
			return ModelRunResult(
				model=model,
				run_dir=run_dir,
				run_report=load_json(run_report_path),
				initial_eval_report=initial_eval,
				recover_report=recover_report,
				eval_report=None,
				failed_before_recover=failed_before,
				failed_after_recover=None,
				success=False,
				error=f"eval-after-recover failed (code={eval_proc_2.returncode}): {eval_proc_2.stderr.strip()}",
			)

		failed_after = count_missing_in_summary(summary_path)

		return ModelRunResult(
			model=model,
			run_dir=run_dir,
			run_report=load_json(run_report_path),
			initial_eval_report=initial_eval,
			recover_report=recover_report,
			eval_report=load_json(eval_report_path),
			failed_before_recover=failed_before,
			failed_after_recover=failed_after,
			success=True,
			error=None,
		)
	except Exception as exc:  # noqa: BLE001
		return ModelRunResult(
			model=model,
			run_dir=None,
			run_report=None,
			initial_eval_report=None,
			recover_report=None,
			eval_report=None,
			failed_before_recover=None,
			failed_after_recover=None,
			success=False,
			error=str(exc),
		)


def repair_existing_run(model: str, args: argparse.Namespace) -> ModelRunResult:
	"""Only run eval/recover/eval for an existing run dir."""
	try:
		try:
			run_dir = resolve_existing_run_dir(model, Path(args.dataset).stem, args.output_dir)
		except Exception as exc:  # noqa: BLE001
			return ModelRunResult(
				model=model,
				run_dir=None,
				run_report=None,
				initial_eval_report=None,
				recover_report=None,
				eval_report=None,
				failed_before_recover=None,
				failed_after_recover=None,
				success=False,
				error=f"Existing run not found for repair: {exc}",
			)

		run_report_path = run_dir / "run_report.json"
		summary_path = run_dir / "summary.jsonl"
		if not summary_path.exists():
			return ModelRunResult(
				model=model,
				run_dir=run_dir,
				run_report=load_json(run_report_path) if run_report_path.exists() else None,
				initial_eval_report=None,
				recover_report=None,
				eval_report=None,
				failed_before_recover=None,
				failed_after_recover=None,
				success=False,
				error=f"Missing summary.jsonl in {run_dir}",
			)

		failed_before = count_missing_in_summary(summary_path)
		print(f"[repair] model={model} missing_files_before={failed_before}", flush=True)

		eval_proc = run_eval(run_dir, args.answer_rtol)
		if eval_proc.returncode != 0:
			return ModelRunResult(
				model=model,
				run_dir=run_dir,
				run_report=load_json(run_report_path) if run_report_path.exists() else None,
				initial_eval_report=None,
				recover_report=None,
				eval_report=None,
				failed_before_recover=failed_before,
				failed_after_recover=None,
				success=False,
				error=f"eval-before-recover failed (code={eval_proc.returncode}): {eval_proc.stderr.strip()}",
			)

		initial_eval = load_json(run_dir / "eval_report.json")
		recover_proc = run_recover(
			run_dir=run_dir,
			bench_file=args.dataset,
			max_tokens=args.max_tokens,
			maximum_retries=args.maximum_retries,
			timeout_s=args.timeout_s,
		)
		if recover_proc.returncode != 0:
			return ModelRunResult(
				model=model,
				run_dir=run_dir,
				run_report=load_json(run_report_path) if run_report_path.exists() else None,
				initial_eval_report=initial_eval,
				recover_report=None,
				eval_report=None,
				failed_before_recover=failed_before,
				failed_after_recover=None,
				success=False,
				error=f"recover failed (code={recover_proc.returncode}): {recover_proc.stderr.strip()}",
			)

		eval_proc_2 = run_eval(run_dir, args.answer_rtol)
		if eval_proc_2.returncode != 0:
			return ModelRunResult(
				model=model,
				run_dir=run_dir,
				run_report=load_json(run_report_path) if run_report_path.exists() else None,
				initial_eval_report=initial_eval,
				recover_report=load_json(run_dir / "recover_report.json") if (run_dir / "recover_report.json").exists() else None,
				eval_report=None,
				failed_before_recover=failed_before,
				failed_after_recover=None,
				success=False,
				error=f"eval-after-recover failed (code={eval_proc_2.returncode}): {eval_proc_2.stderr.strip()}",
			)

		failed_after = count_missing_in_summary(summary_path)
		return ModelRunResult(
			model=model,
			run_dir=run_dir,
			run_report=load_json(run_report_path) if run_report_path.exists() else None,
			initial_eval_report=initial_eval,
			recover_report=load_json(run_dir / "recover_report.json") if (run_dir / "recover_report.json").exists() else None,
			eval_report=load_json(run_dir / "eval_report.json") if (run_dir / "eval_report.json").exists() else None,
			failed_before_recover=failed_before,
			failed_after_recover=failed_after,
			success=True,
			error=None,
		)
	except Exception as exc:  # noqa: BLE001
		return ModelRunResult(
			model=model,
			run_dir=None,
			run_report=None,
			initial_eval_report=None,
			recover_report=None,
			eval_report=None,
			failed_before_recover=None,
			failed_after_recover=None,
			success=False,
			error=str(exc),
		)


def print_summary(results: list[ModelRunResult], benchmark: Path, experiment_name: str) -> None:
	print("\n=== Combined Benchmark Summary ===")
	print(f"benchmark: {benchmark}")
	print(f"experiment_name: {experiment_name}")

	for item in results:
		print(f"\nModel: {item.model}")
		print(f"  success: {item.success}")
		if item.run_dir is not None:
			print(f"  run_dir: {item.run_dir}")

		if item.run_report:
			print(f"  num_questions: {item.run_report.get('num_questions')}")
			print(f"  request_success_count: {item.run_report.get('request_success_count')}")
			print(f"  execution_success_count: {item.run_report.get('execution_success_count')}")
			print(f"  match_count: {item.run_report.get('match_count')}")

		if item.failed_before_recover is not None:
			print(f"  failed_before_recover: {item.failed_before_recover}")
		if item.failed_after_recover is not None:
			print(f"  failed_after_recover: {item.failed_after_recover}")

		if item.recover_report is not None:
			print(f"  recovered_count: {item.recover_report.get('recovered_count')}")

		if item.eval_report:
			answer_match = item.eval_report.get("answer_match", {})
			print(f"  eval_answer_match: {answer_match.get('count')}/{item.eval_report.get('num_questions')}")
			print(f"  eval_answer_match_ratio: {answer_match.get('ratio')}")

		if item.error:
			print(f"  error: {item.error}")


def build_arg_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description="Combined benchmark runner (robust and model-parameterized)")
	parser.add_argument("--model", type=str, required=True, help="Run a single model")
	parser.add_argument("--dataset", type=Path, required=True, help="Benchmark JSONL file")
	parser.add_argument("--output-dir", type=Path, required=True, help="Directory to save the results.")
	parser.add_argument("--max-workers", type=int, default=DEFAULT_MAX_WORKERS)
	parser.add_argument("--timeout-s", type=float, default=DEFAULT_TIMEOUT_S)
	parser.add_argument("--maximum-retries", type=int, default=DEFAULT_MAXIMUM_RETRIES)
	parser.add_argument("--answer-rtol", type=float, default=DEFAULT_ANSWER_RTOL)
	parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
	parser.add_argument(
		"--repair-only",
		action="store_true",
		help="Do not generate; only repair existing run folder (<history>/<model>/<experiment-name>)",
	)
	return parser


def main() -> int:
	args = build_arg_parser().parse_args()

	print("Using Python:", sys.executable)
	print("Benchmark:", args.dataset)
	print("Output dir:", args.output_dir)
	print("Model:", args.model)

	if args.repair_only:
		result = repair_existing_run(args.model, args)
	else:
		result = run_one_model(args.model, args)

	print_summary([result], benchmark=args.dataset, experiment_name=Path(args.dataset).stem)

	return 1 if not result.success else 0


if __name__ == "__main__":
	raise SystemExit(main())
