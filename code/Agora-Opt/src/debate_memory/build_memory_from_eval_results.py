#!/usr/bin/env python3
"""
Build solution memory from evaluation result directories.

Any evaluation directory can be used as input as long as it contains both
`evaluation_results.jsonl` and a `code/` directory. The script extracts problem
descriptions, executable code, and objective values from correct cases and
writes them into the solution-memory store.
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

from .config import find_benchmark_path, get_benchmark_dirs, normalize_dataset_name
from .memory_bank import MemoryBank

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
DEFAULT_BENCHMARKS_DIR = get_benchmark_dirs(PROJECT_ROOT)[0]


def load_evaluation_results(eval_file: str) -> Dict[int, Dict]:
    """Load evaluation results as `{id: {...}}`."""
    results = {}
    if not os.path.exists(eval_file):
        print(f"Warning: evaluation result file not found: {eval_file}")
        return results
    
    with open(eval_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                results[data['id']] = data
    return results


def load_benchmark_data(benchmark_file: str) -> Dict[int, Dict]:
    """Load benchmark data as `{id: {...}}`."""
    data = {}
    if not os.path.exists(benchmark_file):
        print(f"Warning: benchmark file not found: {benchmark_file}")
        return data
    
    with open(benchmark_file, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            if line.strip():
                item = json.loads(line)
                # Prefer an explicit id field, otherwise fall back to the line index.
                problem_id = item.get('id', item.get('problem_id', idx))
                data[problem_id] = item
    return data


def load_solution_code(code_file: str) -> Optional[str]:
    """Load a solution code file."""
    if not os.path.exists(code_file):
        return None
    
    try:
        with open(code_file, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        print(f"Warning: failed to read code file {code_file}: {e}")
        return None


def extract_dataset_name(eval_dir: str) -> Optional[str]:
    """
    Extract the dataset name from an evaluation directory name.

    Example:
        `deepseek-chat_EasyLP_clean_eval_20251024_120712.jsonl` -> `EasyLP`
    """
    dir_name = os.path.basename(eval_dir)
    # Remove the .jsonl suffix if present.
    if dir_name.endswith('.jsonl'):
        dir_name = dir_name[:-6]
    
    # Remove the model name and timestamp.
    parts = dir_name.split('_')
    # Locate the `eval` marker.
    try:
        eval_idx = parts.index('eval')
        # The dataset name should appear before `eval`, after the model name.
        dataset_parts = parts[:eval_idx]
        if len(dataset_parts) > 1:
            return normalize_dataset_name('_'.join(dataset_parts[1:]))
        else:
            return normalize_dataset_name(dataset_parts[0]) if dataset_parts else None
    except ValueError:
        # Fallback for names of the form model_dataset_timestamp.
        if len(parts) >= 3:
            return normalize_dataset_name('_'.join(parts[1:-1]))
        return None


def build_memory_from_eval_result(eval_result_dir: str, benchmarks_dir: str, memory_bank: MemoryBank):
    """
    Build memory from a single evaluation result directory.

    Args:
        eval_result_dir: Directory containing `evaluation_results.jsonl` and `code/`.
        benchmarks_dir: Benchmark dataset directory.
        memory_bank: MemoryBank instance.
    """
    eval_file = os.path.join(eval_result_dir, 'evaluation_results.jsonl')
    code_dir = os.path.join(eval_result_dir, 'code')
    
    if not os.path.exists(eval_file):
        print(f"Warning: skipping {eval_result_dir}: evaluation_results.jsonl not found")
        return 0, 0
    
    # Extract the dataset name.
    dataset_name = extract_dataset_name(eval_result_dir)
    if not dataset_name:
        print(f"Warning: skipping {eval_result_dir}: failed to extract dataset name")
        return 0, 0
    
    benchmark_file = os.path.join(benchmarks_dir, f"{dataset_name}.jsonl")
    if not os.path.exists(benchmark_file):
        try:
            benchmark_file = str(find_benchmark_path(PROJECT_ROOT, dataset_name))
        except FileNotFoundError:
            pass
    if not os.path.exists(benchmark_file):
        print(f"Warning: skipping {eval_result_dir}: benchmark file not found: {benchmark_file}")
        return 0, 0
    
    print(f"Processing dataset: {dataset_name}")
    print(f"  evaluation results: {eval_file}")
    print(f"  benchmark file: {benchmark_file}")
    print(f"  code directory: {code_dir}")
    
    # Load all required inputs.
    eval_results = load_evaluation_results(eval_file)
    benchmark_data = load_benchmark_data(benchmark_file)
    
    added_count = 0
    skipped_count = 0
    
    # Process each correct case.
    for problem_id, eval_result in eval_results.items():
        # Only keep correct cases.
        if not eval_result.get('is_correct', False):
            skipped_count += 1
            continue
        
        # Recover the problem description.
        if problem_id not in benchmark_data:
            print(f"  Warning: skipping ID {problem_id}: missing from benchmark file")
            skipped_count += 1
            continue
        
        benchmark_item = benchmark_data[problem_id]
        # Support both `description` and `en_question`.
        description = benchmark_item.get('description', '') or benchmark_item.get('en_question', '')
        
        if not description:
            print(f"  Warning: skipping ID {problem_id}: missing problem description")
            skipped_count += 1
            continue
        
        # Load the solution code.
        code_file = os.path.join(code_dir, f"problem_{problem_id}.py")
        solution_code = load_solution_code(code_file)
        
        if not solution_code:
            print(f"  Warning: skipping ID {problem_id}: code file missing or unreadable")
            skipped_count += 1
            continue
        
        # Recover the objective value.
        objective_value = eval_result.get('predicted_objective')
        if objective_value is None:
            # Fall back to the benchmark answer fields if needed.
            answer_str = benchmark_item.get('answer', '') or benchmark_item.get('en_answer', '')
            try:
                objective_value = float(answer_str)
            except:
                print(f"  Warning: skipping ID {problem_id}: objective value unavailable")
                skipped_count += 1
                continue
        
        # Build metadata for the stored case.
        ground_truth = benchmark_item.get('answer', '') or benchmark_item.get('en_answer', '')
        metadata = {
            'source': 'eval_results',
            'dataset': dataset_name,
            'eval_dir': os.path.basename(eval_result_dir),
            'execution_status': eval_result.get('execution_status', 'unknown'),
            'ground_truth': ground_truth,
        }
        
        # Do not deduplicate across datasets; the same problem_id may appear in multiple benchmarks.
        
        # Add the case to the memory bank.
        try:
            memory_bank.add_case(
                problem_id=problem_id,
                problem_desc=description,
                solution_code=solution_code,
                objective_value=float(objective_value),
                is_correct=True,
                metadata=metadata
            )
            added_count += 1
        except Exception as e:
            print(f"  Error: failed to add ID {problem_id}: {e}")
            skipped_count += 1
    
    print(f"  added cases: {added_count}")
    print(f"  skipped cases: {skipped_count}")
    print()
    
    return added_count, skipped_count


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Build a memory bank from evaluation results")
    parser.add_argument('--eval_dirs', type=str, nargs='+', required=True,
                       help='Evaluation result directories containing evaluation_results.jsonl and code/')
    parser.add_argument('--benchmarks_dir', type=str,
                       default=str(DEFAULT_BENCHMARKS_DIR),
                       help='Benchmark dataset directory')
    parser.add_argument('--memory_dir', type=str,
                       default=str(PROJECT_ROOT / "memory_storage"),
                       help='Memory storage directory')
    parser.add_argument('--clear', action='store_true',
                       help='Clear the existing memory store before building')
    
    args = parser.parse_args()
    
    # Validate input directories.
    if not os.path.exists(args.benchmarks_dir):
        print(f"Error: benchmark directory does not exist: {args.benchmarks_dir}")
        sys.exit(1)
    
    # Clear the memory store if requested.
    if args.clear:
        if os.path.exists(args.memory_dir):
            import shutil
            print(f"Clearing existing memory store: {args.memory_dir}")
            shutil.rmtree(args.memory_dir)
            print()
    
    # Initialize the memory bank.
    print("="*70)
    print("Building Memory Bank from Evaluation Results")
    print("="*70)
    print()
    
    memory_bank = MemoryBank(memory_dir=args.memory_dir)
    print(f"Current memory size: {memory_bank.case_count} cases")
    print()
    
    # Process each evaluation directory.
    total_added = 0
    total_skipped = 0
    
    for eval_dir in args.eval_dirs:
        if not os.path.exists(eval_dir):
            print(f"Warning: skipping missing directory: {eval_dir}")
            continue
        
        added, skipped = build_memory_from_eval_result(
            eval_dir, args.benchmarks_dir, memory_bank
        )
        total_added += added
        total_skipped += skipped
    
    # Refresh the case count.
    memory_bank.case_count = memory_bank._count_cases()
    
    print("="*70)
    print("Memory Bank Build Complete")
    print("="*70)
    print(f"Total added: {total_added} cases")
    print(f"Total skipped: {total_skipped} cases")
    print(f"Final memory size: {memory_bank.case_count} cases")
    print()
    print(f"Memory location: {args.memory_dir}")
    print(f"   - cases.jsonl: {os.path.join(args.memory_dir, 'cases.jsonl')}")
    print(f"   - index/: {os.path.join(args.memory_dir, 'index')}")
    print("="*70)


if __name__ == "__main__":
    main()
