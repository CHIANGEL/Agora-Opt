#!/usr/bin/env python3
"""
Run debates between two models using memory-augmented single generations.

This script automatically locates the latest initial-solution files for the
specified models, runs the parallel debate workflow from `simple_rag/debate.py`,
and then evaluates the consensus solutions with `execute.py`.

Example:
    python run_memory_debate.py \
        --datasets ComplexLP EasyLP \
        --max_rounds 3 \
        --debate_workers 16 \
        --execute_workers 128
"""
from __future__ import annotations

import argparse
import datetime as dt
import glob
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .config import normalize_dataset_name

SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = SRC_DIR.parent
MONOREPO_ROOT = PROJECT_ROOT.parent
STANDARD_RESULTS_ROOT = PROJECT_ROOT.parent.parent / "results" / "Agora-Opt"
DEFAULT_RESULTS_DIR = STANDARD_RESULTS_ROOT / "generation"
DEFAULT_OUTPUT_ROOT = STANDARD_RESULTS_ROOT / "debate"
DEFAULT_DEBATE_SCRIPT = MONOREPO_ROOT / "simple_rag" / "debate.py"
DEFAULT_EXECUTE_SCRIPT = PROJECT_ROOT / "scripts" / "execute.py"
DEFAULT_DEBATE_MEMORY_DIR = PROJECT_ROOT / "debate_memory_storage"
DEBATE_MEMORY_HEADER = "# Debate Memory Insights"

from .memory_bank import MemoryBank


def format_debate_memory_context(cases: List[Dict]) -> str:
    if not cases:
        return ""
    lines = [DEBATE_MEMORY_HEADER, ""]
    for idx, item in enumerate(cases, 1):
        case = item["case"]
        score = item.get("score", 0.0)
        metadata = case.get("metadata", {})
        dataset = metadata.get("dataset", "unknown")
        summary = metadata.get("summary", {}).get("summary")
        lines.append(f"## Case {idx} (similarity {score:.3f}, dataset {dataset})")
        description = case.get("description", "").strip()
        if description:
            snippet = description if len(description) <= 800 else description[:800] + "\n..."
            lines.append(snippet)
        if summary:
            lines.append("Summary: " + summary)
        lines.append("---")
    return "\n".join(lines).strip()


def build_debate_memory_contexts(
    files: List[str],
    debate_memory: MemoryBank,
    dataset: str,
    top_k: int,
) -> Dict[int, str]:
    contexts: Dict[int, str] = {}
    if debate_memory is None or top_k <= 0:
        return contexts
    for file_path in files:
        with open(file_path, "r", encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                data = json.loads(line)
                problem_id = data.get("id")
                if problem_id is None or problem_id in contexts:
                    continue
                description = data.get("description", "")
                if not description.strip():
                    contexts[problem_id] = ""
                    continue
                cases = debate_memory.retrieve_similar_cases(
                    description,
                    top_k=top_k,
                    preferred_dataset=dataset,
                )
                contexts[problem_id] = format_debate_memory_context(cases)
    return contexts


def maybe_enrich_generation_file(
    source_path: str,
    destination_path: str,
    contexts: Dict[int, str],
) -> str:
    if not contexts:
        return source_path
    changed = False
    enriched_lines: List[str] = []
    with open(source_path, "r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            data = json.loads(line)
            pid = data.get("id")
            context = contexts.get(pid)
            if context:
                data["description"] = f"{data.get('description', '').strip()}\n\n{context}"
                changed = True
            enriched_lines.append(json.dumps(data, ensure_ascii=False))
    if not changed:
        return source_path
    with open(destination_path, "w", encoding="utf-8") as fh:
        for entry in enriched_lines:
            fh.write(entry + "\n")
    return destination_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Parallel debate runner for memory-enhanced single generations"
    )
    parser.add_argument(
        "--modelA",
        type=str,
        default="gpt-4o",
        help="First model in the debate (default: gpt-4o)",
    )
    parser.add_argument(
        "--modelB",
        type=str,
        default="deepseek-chat",
        help="Second model in the debate (default: deepseek-chat)",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default=str(DEFAULT_RESULTS_DIR),
        help="Directory that stores initial-solution JSONL files",
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=None,
        help="Datasets to debate. If omitted, auto-detect common datasets.",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default=str(DEFAULT_OUTPUT_ROOT),
        help="Root directory to store debate/eval outputs",
    )
    parser.add_argument(
        "--debate_script",
        type=str,
        default=str(DEFAULT_DEBATE_SCRIPT),
        help="Path to simple_rag/debate.py (override if needed)",
    )
    parser.add_argument(
        "--execute_script",
        type=str,
        default=str(DEFAULT_EXECUTE_SCRIPT),
        help="Path to debate_with_memory/execute.py (override if needed)",
    )
    parser.add_argument(
        "--max_rounds",
        type=int,
        default=3,
        help="Maximum number of debate rounds (default: 3)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.01,
        help="Temperature for debate LLM calls (default: 0.01)",
    )
    parser.add_argument(
        "--debate_workers",
        type=int,
        default=16,
        help="Parallel workers for debate (ThreadPool inside debate.py)",
    )
    parser.add_argument(
        "--execute_workers",
        type=int,
        default=128,
        help="Parallel workers for execute.py evaluation",
    )
    parser.add_argument(
        "--max_problems",
        type=int,
        default=None,
        help="Optional cap on number of problems per dataset",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.05,
        help="Relative tolerance for evaluation accuracy comparison",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=90,
        help="Timeout (seconds) for executing consensus code",
    )
    parser.add_argument(
        "--relative_tolerance",
        action="store_true",
        help="Pass --use_relative_tolerance to execute.py",
    )
    parser.add_argument(
        "--save_execution_stdout",
        action="store_true",
        help="Store stdout/stderr for consensus executions",
    )
    parser.add_argument(
        "--execute_memory_dir",
        type=str,
        default=None,
        help="Optional memory_storage directory forwarded to execute.py during consensus evaluation.",
    )
    parser.add_argument(
        "--execute_debug_memory_path",
        type=str,
        default=None,
        help="Optional debug_memory.jsonl path forwarded to execute.py during consensus evaluation.",
    )
    parser.add_argument(
        "--execute_disable_debug_memory",
        action="store_true",
        help="Pass --disable_debug_memory to execute.py during consensus evaluation.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Only print the planned actions without running debate/eval",
    )
    parser.add_argument(
        "--debate_memory_dir",
        type=str,
        default=str(DEFAULT_DEBATE_MEMORY_DIR),
        help="Directory containing debate memory cases for prompt augmentation",
    )
    parser.add_argument(
        "--debate_memory_top_k",
        type=int,
        default=2,
        help="How many debate memory cases to retrieve per problem",
    )
    parser.add_argument(
        "--disable_debate_memory",
        action="store_true",
        help="Skip retrieval even if debate memory directory exists",
    )
    parser.add_argument(
        "--embedding_model",
        type=str,
        default=None,
        help="Embedding model name or local path (default: BAAI/bge-small-en-v1.5). "
             "Use local path to avoid Hugging Face downloads, or set HF_HUB_OFFLINE=1 environment variable.",
    )
    return parser.parse_args()


def normalize_dataset_list(raw_list: Optional[List[str]]) -> Optional[List[str]]:
    """Split comma-separated values and strip whitespace."""
    if not raw_list:
        return None
    datasets: List[str] = []
    for item in raw_list:
        parts = [part.strip() for part in item.split(",") if part.strip()]
        datasets.extend(normalize_dataset_name(part) for part in parts)
    return datasets or None


def collect_runs(results_dir: str, model: str) -> Dict[str, List[Tuple[str, str]]]:
    """
    Return mapping dataset -> list of (timestamp, path) sorted ascending.
    Skips evaluation artifacts (suffixes containing '_eval').
    """
    pattern = os.path.join(results_dir, f"{model}_*.jsonl")
    regex = re.compile(rf"{re.escape(model)}_(.+)_(\d{{8}}_\d{{6}})\.jsonl$")
    runs: Dict[str, List[Tuple[str, str]]] = {}

    for path in glob.glob(pattern):
        base = os.path.basename(path)
        match = regex.match(base)
        if not match:
            continue
        dataset = normalize_dataset_name(match.group(1))
        if "_eval" in dataset:
            continue
        timestamp = match.group(2)
        runs.setdefault(dataset, []).append((timestamp, path))

    for dataset in runs:
        runs[dataset].sort()  # chronological

    return runs


def pick_latest(runs: Dict[str, List[Tuple[str, str]]], dataset: str) -> Optional[str]:
    """Return latest file path for dataset if available."""
    entries = runs.get(dataset)
    if not entries:
        return None
    return entries[-1][1]


def stream_command(cmd: List[str], cwd: str, log_path: str) -> None:
    """Run a subprocess, streaming output to stdout and a log file."""
    print(f"\n▶ Running: {' '.join(cmd)}", flush=True)
    print(f"   cwd: {cwd}", flush=True)
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    with open(log_path, "w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            cmd,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        assert process.stdout is not None  # for type checkers
        for line in process.stdout:
            print(line, end="", flush=True)
            log_file.write(line)
            log_file.flush()
        return_code = process.wait()

    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, cmd)


def load_eval_report(report_path: str) -> Optional[Dict]:
    if not os.path.exists(report_path):
        return None
    with open(report_path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def ensure_script(path: str, description: str) -> None:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{description} not found: {path}")


def main() -> None:
    args = parse_args()
    args.datasets = normalize_dataset_list(args.datasets)
    args.output_root = os.path.abspath(args.output_root)
    args.results_dir = os.path.abspath(args.results_dir)

    debate_memory_bank: Optional[MemoryBank] = None
    if not args.disable_debate_memory and args.debate_memory_dir:
        debate_memory_path = Path(args.debate_memory_dir)
        if debate_memory_path.exists():
            try:
                embedding_model = args.embedding_model if args.embedding_model else "BAAI/bge-small-en-v1.5"
                debate_memory_bank = MemoryBank(
                    memory_dir=str(debate_memory_path),
                    embedding_model=embedding_model
                )
            except Exception as exc:  # noqa: BLE001
                print(f"⚠️  Warning: failed to load debate memory from {debate_memory_path}: {exc}")
        else:
            print(f"ℹ️  Debate memory directory not found: {debate_memory_path} (skipping context retrieval)")

    ensure_script(args.debate_script, "Debate script")
    ensure_script(args.execute_script, "Execute script")

    modelA_runs = collect_runs(args.results_dir, args.modelA)
    modelB_runs = collect_runs(args.results_dir, args.modelB)

    if args.datasets:
        datasets = args.datasets
    else:
        datasets = sorted(set(modelA_runs.keys()) & set(modelB_runs.keys()))

    if not datasets:
        print("❌ No common datasets with available runs were found.")
        sys.exit(1)

    print("=" * 80)
    print("🧠 Memory-Based Debate Runner")
    print("=" * 80)
    print(f"Model A: {args.modelA}")
    print(f"Model B: {args.modelB}")
    print(f"Datasets: {', '.join(datasets)}")
    print(f"Results dir: {args.results_dir}")
    print(f"Output root: {args.output_root}")
    print(f"Debate workers: {args.debate_workers} (parallel)")
    print("=" * 80)

    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset_summaries: List[Dict] = []
    processed = 0

    for dataset in datasets:
        file_a = pick_latest(modelA_runs, dataset)
        file_b = pick_latest(modelB_runs, dataset)

        if not file_a or not file_b:
            print(f"⚠️  Skipping {dataset}: missing runs for one of the models.")
            dataset_summaries.append(
                {
                    "dataset": dataset,
                    "status": "missing_runs",
                    "modelA_file": file_a,
                    "modelB_file": file_b,
                }
            )
            continue

        run_dir = os.path.join(
            args.output_root,
            dataset,
            f"{timestamp}_{args.modelA}_vs_{args.modelB}",
        )
        os.makedirs(run_dir, exist_ok=True)

        print(f"\n{'=' * 80}")
        print(f"🚀 Dataset: {dataset}")
        print(f"   Model A file: {file_a}")
        print(f"   Model B file: {file_b}")
        print(f"   Output dir: {run_dir}")
        print(f"{'=' * 80}")

        file_a_for_debate = file_a
        file_b_for_debate = file_b
        if not args.dry_run and debate_memory_bank and args.debate_memory_top_k > 0:
            contexts = build_debate_memory_contexts(
                [file_a, file_b], debate_memory_bank, dataset, args.debate_memory_top_k
            )
            if any(contexts.values()):
                print("  🧠 Injecting debate memory context into prompts")
                enriched_a = os.path.join(
                    run_dir, f"{os.path.basename(file_a)}.debate_memory.jsonl"
                )
                enriched_b = os.path.join(
                    run_dir, f"{os.path.basename(file_b)}.debate_memory.jsonl"
                )
                file_a_for_debate = maybe_enrich_generation_file(file_a, enriched_a, contexts)
                file_b_for_debate = maybe_enrich_generation_file(file_b, enriched_b, contexts)

        if args.dry_run:
            print("Dry-run mode → skipping actual execution.")
            dataset_summaries.append(
                {
                    "dataset": dataset,
                    "status": "dry_run",
                    "debate_dir": run_dir,
                    "modelA_file": file_a,
                    "modelB_file": file_b,
                }
            )
            continue

        # 1) Run debate
        debate_cmd = [
            sys.executable,
            "-u",
            args.debate_script,
            "--resultA",
            file_a_for_debate,
            "--resultB",
            file_b_for_debate,
            "--modelA",
            args.modelA,
            "--modelB",
            args.modelB,
            "--save_dir",
            run_dir,
            "--max_rounds",
            str(args.max_rounds),
            "--temperature",
            str(args.temperature),
            "--num_workers",
            str(args.debate_workers),
        ]
        if args.max_problems is not None:
            debate_cmd.extend(["--max_problems", str(args.max_problems)])

        debate_log = os.path.join(run_dir, "debate.log")
        stream_command(debate_cmd, cwd=str(MONOREPO_ROOT), log_path=debate_log)

        consensus_file = os.path.join(
            run_dir, f"consensus_{args.modelA}_vs_{args.modelB}.jsonl"
        )
        if not os.path.exists(consensus_file):
            raise FileNotFoundError(
                f"Consensus file not found after debate: {consensus_file}"
            )

        # 2) Evaluate consensus
        eval_dir = os.path.join(run_dir, "eval_consensus")
        eval_cmd = [
            sys.executable,
            "-u",
            args.execute_script,
            "--input_file",
            consensus_file,
            "--output_dir",
            eval_dir,
            "--timeout",
            str(args.timeout),
            "--tolerance",
            str(args.tolerance),
            "--num_workers",
            str(args.execute_workers),
        ]
        if args.relative_tolerance:
            eval_cmd.append("--use_relative_tolerance")
        if args.save_execution_stdout:
            eval_cmd.append("--save_output")
        if args.execute_memory_dir:
            eval_cmd.extend(["--memory_dir", args.execute_memory_dir])
        if args.execute_debug_memory_path:
            eval_cmd.extend(["--debug_memory_path", args.execute_debug_memory_path])
        if args.execute_disable_debug_memory:
            eval_cmd.append("--disable_debug_memory")
        if args.embedding_model:
            eval_cmd.extend(["--embedding_model", args.embedding_model])

        eval_log = os.path.join(run_dir, "evaluate.log")
        stream_command(eval_cmd, cwd=str(PROJECT_ROOT), log_path=eval_log)

        report_path = os.path.join(eval_dir, "evaluation_report.json")
        report = load_eval_report(report_path)
        if report is None:
            raise FileNotFoundError(f"Missing evaluation report: {report_path}")

        dataset_summaries.append(
            {
                "dataset": dataset,
                "status": "completed",
                "debate_dir": run_dir,
                "accuracy": report.get("accuracy"),
                "correct": report.get("correct"),
                "total": report.get("total_problems"),
                "report_path": report_path,
            }
        )
        processed += 1

    print("\n" + "=" * 80)
    print("📊 Debate + Evaluation Summary")
    print("=" * 80)
    for item in dataset_summaries:
        dataset = item["dataset"]
        status = item["status"]
        if status == "completed":
            accuracy = item.get("accuracy")
            correct = item.get("correct")
            total = item.get("total")
            print(
                f"{dataset:25s} → accuracy {accuracy:.2%} ({correct}/{total}) | dir: {item['debate_dir']}"
            )
        elif status == "dry_run":
            print(f"{dataset:25s} → dry run (planned dir: {item['debate_dir']})")
        else:
            print(f"{dataset:25s} → {status} (A={item.get('modelA_file')}, B={item.get('modelB_file')})")

    print("=" * 80)
    if not args.dry_run and processed == 0:
        sys.exit("No datasets were processed successfully.")


if __name__ == "__main__":
    main()
