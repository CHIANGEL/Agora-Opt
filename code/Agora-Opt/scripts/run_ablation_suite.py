#!/usr/bin/env python3
"""
Run a suite of ablation experiments (generation + evaluation) and summarise results.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
STANDARD_RESULTS_ROOT = PROJECT_ROOT.parent.parent / "results" / "Agora-Opt"
GENERATE_SCRIPT = PROJECT_ROOT / "scripts" / "generate_with_memory.py"
EXECUTE_SCRIPT = PROJECT_ROOT / "scripts" / "execute.py"
PYTHON_BIN = os.environ.get("PYTHON_BIN", sys.executable)


@dataclass
class Variant:
    name: str
    description: str
    overrides: Dict[str, object]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run generate+evaluate ablations and emit a summary table."
    )
    parser.add_argument("--model", type=str, default="gpt-4o", help="LLM to query.")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["IndustryOR", "ComplexLP"],
        help="Datasets to evaluate (space-separated, omit .jsonl).",
    )
    parser.add_argument("--temperature", type=float, default=0.01)
    parser.add_argument(
        "--max_problems",
        type=int,
        default=None,
        help="Limit number of problems per dataset (omit for full set).",
    )
    parser.add_argument("--memory_dir", type=str, default="memory_storage")
    parser.add_argument(
        "--memory_top_k",
        type=int,
        default=3,
        help="Base episodic memory retrieval count for the full variant.",
    )
    parser.add_argument(
        "--max_retries",
        type=int,
        default=5,
        help="Base retry budget for the full variant.",
    )
    parser.add_argument(
        "--debug_case_top_k",
        type=int,
        default=3,
        help="Base debug-case retrieval count.",
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=64,
        help="Workers for generation (passed to --parallel).",
    )
    parser.add_argument(
        "--execution_timeout",
        type=int,
        default=90,
        help="Timeout per execution attempt in generate_with_memory.",
    )
    parser.add_argument(
        "--debug_memory_path",
        type=str,
        default="memory_storage/debug_memory.jsonl",
        help="Path to debug memory JSONL.",
    )
    parser.add_argument(
        "--debug_case_dir",
        type=str,
        default="debug_case_memory",
        help="Directory containing consolidated debug-case memory.",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default=str(STANDARD_RESULTS_ROOT / "ablations"),
        help="Root folder for storing ablation artefacts.",
    )
    parser.add_argument(
        "--eval_timeout",
        type=int,
        default=90,
        help="Timeout for scripts/execute.py.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=64,
        help="ProcessPool workers for evaluation.",
    )
    parser.add_argument("--tolerance", type=float, default=0.05)
    parser.add_argument(
        "--relative_tolerance",
        action="store_true",
        help="Use relative tolerance in evaluation.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print commands without executing or aggregating results.",
    )
    return parser.parse_args()


def build_variants(args: argparse.Namespace) -> List[Variant]:
    base = {
        "memory_top_k": args.memory_top_k,
        "use_llm_refinement": True,
        "debug_case_memory_top_k": args.debug_case_top_k,
        "max_retries": args.max_retries,
        "auto_debug": True,
    }
    return [
        Variant(
            name="full_system",
            description="All helpers enabled (reference).",
            overrides={**base},
        ),
        Variant(
            name="no_llm_refine",
            description="Skip LLM summarisation of retrieved cases.",
            overrides={**base, "use_llm_refinement": False},
        ),
        Variant(
            name="no_debug_case_memory",
            description="Disable historical debug-case retrieval.",
            overrides={**base, "debug_case_memory_top_k": 0},
        ),
        Variant(
            name="no_self_healing",
            description="Single attempt (max_retries=1) but still executes locally once.",
            overrides={**base, "max_retries": 1},
        ),
        Variant(
            name="no_memory",
            description="Disable episodic retrieval, keep retries on.",
            overrides={**base, "memory_top_k": 0, "use_llm_refinement": False},
        ),
        Variant(
            name="vanilla_llm",
            description="Pure single-shot LLM (no memory, no auto-debug).",
            overrides={
                **base,
                "memory_top_k": 0,
                "use_llm_refinement": False,
                "debug_case_memory_top_k": 0,
                "max_retries": 1,
                "auto_debug": False,
            },
        ),
    ]


def run_command(cmd: Sequence[str], dry_run: bool = False) -> None:
    pretty = " ".join(shlex.quote(part) for part in cmd)
    print(f"  → {pretty}")
    if dry_run:
        return
    subprocess.run(cmd, check=True)


def compute_attempt_stats(path: Path) -> Tuple[float, int]:
    if not path.exists():
        return 0.0, 0
    total = 0
    total_attempts = 0
    multi_attempt = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            attempts = record.get("total_attempts", 1)
            total_attempts += attempts
            total += 1
            if attempts > 1:
                multi_attempt += 1
    avg = (total_attempts / total) if total else 0.0
    return avg, multi_attempt


def format_percent(value: float) -> str:
    return f"{value * 100:.1f}%"


def build_generate_args(
    dataset: str,
    output_file: Path,
    debug_dir: Path,
    args: argparse.Namespace,
    cfg: Dict[str, object],
) -> List[str]:
    cmd = [
        os.fspath(GENERATE_SCRIPT),
        "--dataset",
        dataset,
        "--model",
        args.model,
        "--temperature",
        str(args.temperature),
        "--output",
        os.fspath(output_file),
        "--memory_dir",
        os.fspath(Path(args.memory_dir).resolve()),
        "--parallel",
        str(args.parallel),
        "--execution_timeout",
        str(args.execution_timeout),
        "--debug_memory_path",
        os.fspath(Path(args.debug_memory_path).resolve()),
        "--debug_case_memory_dir",
        os.fspath(Path(args.debug_case_dir).resolve()),
        "--debug_case_memory_top_k",
        str(int(cfg.get("debug_case_memory_top_k", 0))),
        "--memory_top_k",
        str(int(cfg.get("memory_top_k", 0))),
        "--max_retries",
        str(int(cfg.get("max_retries", 1))),
    ]
    if args.max_problems:
        cmd += ["--max_problems", str(args.max_problems)]
    if cfg.get("use_llm_refinement"):
        cmd.append("--use_llm_refinement")
    if not cfg.get("filter_perfect", True):
        cmd.append("--no_filter_perfect")
    if not cfg.get("auto_debug", True):
        cmd.append("--no_auto_debug")
    if debug_dir:
        cmd += ["--debug_output_dir", os.fspath(debug_dir)]
    return [os.fspath(part) for part in cmd]


def build_execute_args(input_file: Path, output_dir: Path, args: argparse.Namespace) -> List[str]:
    cmd = [
        os.fspath(EXECUTE_SCRIPT),
        "--input_file",
        os.fspath(input_file),
        "--output_dir",
        os.fspath(output_dir),
        "--timeout",
        str(args.eval_timeout),
        "--tolerance",
        str(args.tolerance),
        "--num_workers",
        str(args.num_workers),
        "--memory_dir",
        os.fspath(Path(args.memory_dir).resolve()),
        "--debug_memory_path",
        os.fspath(Path(args.debug_memory_path).resolve()),
    ]
    if args.relative_tolerance:
        cmd.append("--use_relative_tolerance")
    return cmd


def summarise_records(records: List[Dict], summary_path: Path) -> None:
    if not records:
        return
    md_lines = [
        "| Dataset | Variant | Accuracy | Correct/Total | Exec Err % | Timeout % | No-Code % | Avg Attempts | Notes |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    csv_lines = [
        "dataset,variant,accuracy,correct,total,exec_error_pct,timeout_pct,no_code_pct,avg_attempts,notes"
    ]
    for record in records:
        dataset = record["dataset"]
        variant = record["variant"]
        report = record["report"]
        status_counts = report.get("status_counts", {})
        total = report.get("total_problems", 0)
        accuracy_pct = format_percent(report.get("accuracy", 0.0))
        correct = report.get("correct", 0)
        exec_err_pct = (
            (status_counts.get("execution_error", 0) / total) if total else 0.0
        )
        timeout_pct = (status_counts.get("timeout", 0) / total) if total else 0.0
        no_code_pct = (status_counts.get("no_code", 0) / total) if total else 0.0
        avg_attempts = record.get("avg_attempts", 0.0)
        notes = record["notes"]
        md_lines.append(
            f"| {dataset} | {variant} | {accuracy_pct} | {correct}/{total} | "
            f"{exec_err_pct*100:.1f}% | {timeout_pct*100:.1f}% | {no_code_pct*100:.1f}% | "
            f"{avg_attempts:.2f} | {notes} |"
        )
        safe_notes = notes.replace('"', '""')
        csv_lines.append(
            f"{dataset},{variant},{report.get('accuracy',0.0):.4f},{correct},{total},"
            f"{exec_err_pct:.4f},{timeout_pct:.4f},{no_code_pct:.4f},{avg_attempts:.4f},\"{safe_notes}\""
        )
    summary_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    csv_path = summary_path.with_suffix(".csv")
    csv_path.write_text("\n".join(csv_lines) + "\n", encoding="utf-8")
    print(f"\n✅ Summary table written to: {summary_path}")
    print(f"📄 CSV export written to: {csv_path}")


def main() -> None:
    args = parse_args()
    variants = build_variants(args)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root = Path(args.output_root).resolve() / timestamp
    if not args.dry_run:
        run_root.mkdir(parents=True, exist_ok=True)

    print("========================================")
    print("Ablation Runner")
    print("========================================")
    print(f"Model: {args.model}")
    print(f"Datasets: {', '.join(args.datasets)}")
    print(f"Output root: {run_root if not args.dry_run else args.output_root}")
    print(f"Dry run: {args.dry_run}")
    print("========================================\n")

    records: List[Dict] = []
    for dataset in args.datasets:
        print(f"Dataset: {dataset}")
        for variant in variants:
            cfg = variant.overrides
            variant_name = variant.name
            print(f"  Variant: {variant_name} – {variant.description}")
            dataset_slug = dataset.replace("/", "_")
            gen_output = (
                run_root / f"{dataset_slug}_{variant_name}.jsonl"
                if not args.dry_run
                else Path(f"{dataset_slug}_{variant_name}.jsonl")
            )
            debug_dir = (
                run_root / "debug" / dataset_slug / variant_name
                if not args.dry_run
                else Path(f"debug/{dataset_slug}/{variant_name}")
            )
            eval_dir = (
                run_root / f"{dataset_slug}_{variant_name}_eval"
                if not args.dry_run
                else Path(f"{dataset_slug}_{variant_name}_eval")
            )
            if not args.dry_run:
                debug_dir.mkdir(parents=True, exist_ok=True)
            gen_cmd = [PYTHON_BIN] + build_generate_args(
                dataset, gen_output, debug_dir, args, cfg
            )
            run_command(gen_cmd, dry_run=args.dry_run)

            exec_cmd = [
                PYTHON_BIN,
            ] + build_execute_args(gen_output, eval_dir, args)
            run_command(exec_cmd, dry_run=args.dry_run)

            if args.dry_run:
                continue

            report_path = eval_dir / "evaluation_report.json"
            if not report_path.exists():
                raise FileNotFoundError(
                    f"Missing evaluation report for {dataset} / {variant_name}: {report_path}"
                )
            with report_path.open("r", encoding="utf-8") as handle:
                report = json.load(handle)
            avg_attempts, _ = compute_attempt_stats(gen_output)
            records.append(
                {
                    "dataset": dataset,
                    "variant": variant_name,
                    "report": report,
                    "avg_attempts": avg_attempts,
                    "notes": variant.description,
                }
            )
        print("")

    if args.dry_run:
        print("Dry run completed. No commands were executed.")
        return

    summary_path = run_root / "ablation_summary.md"
    summarise_records(records, summary_path)


if __name__ == "__main__":
    main()
