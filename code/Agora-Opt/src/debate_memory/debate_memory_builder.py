"""
Build a debate-specific memory bank from historical debate runs.

This scans existing debate result directories such as
`./results/Agora-Opt/debate/<dataset>/<timestamp>_<modelA>_vs_<modelB>/`
directories, identifies problems where the two single generations disagreed yet
the debate converged to a correct consensus, summarizes the key reconciliation
insights (optionally via an LLM), and stores the cases inside a dedicated
`MemoryBank` directory (default: ./debate_memory_storage).
"""

from __future__ import annotations

import argparse
import json
import math
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from tqdm import tqdm

from .llm import get_response
from .memory_bank import MemoryBank

PKG_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PKG_DIR.parent.parent
DEFAULT_RUNS_ROOT = PROJECT_ROOT.parent.parent / "results" / "Agora-Opt" / "debate"
DEFAULT_DEBATE_MEMORY_DIR = PROJECT_ROOT / "debate_memory_storage"


@dataclass
class DebateCaseInput:
    dataset: str
    problem_id: int
    description: str
    final_code: str
    final_result: Optional[float]
    debate_rounds: List[Dict]
    modelA: str
    modelB: str
    run_dir: Path
    ground_truth: Optional[str]
    initial_A_result: Optional[float]
    initial_B_result: Optional[float]
    evaluation: Dict
    metadata: Dict


def load_jsonl(path: Path) -> List[Dict]:
    if not path.exists():
        return []
    data: List[Dict] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return data


def float_or_none(value) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (ValueError, TypeError):
        return None


def has_disagreement(entry: Dict, tolerance: float) -> bool:
    a = float_or_none(entry.get("initial_A_result"))
    b = float_or_none(entry.get("initial_B_result"))
    if a is None or b is None:
        return True
    return abs(a - b) > tolerance


def summarize_rounds(rounds: List[Dict], max_chars: int = 2000) -> str:
    if not rounds:
        return ""
    lines: List[str] = []
    for rnd in rounds:
        round_idx = rnd.get("round")
        res_a = rnd.get("result_A")
        res_b = rnd.get("result_B")
        status_a = rnd.get("status_A")
        status_b = rnd.get("status_B")
        analysis_a = (rnd.get("analysis_A") or "").strip()
        analysis_b = (rnd.get("analysis_B") or "").strip()
        lines.append(
            f"Round {round_idx}: A={res_a} ({status_a}), B={res_b} ({status_b})"
        )
        if analysis_a:
            lines.append(f"Model A analysis:\n{analysis_a}")
        if analysis_b:
            lines.append(f"Model B analysis:\n{analysis_b}")
        lines.append("")
    text = "\n".join(lines).strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 200] + "\n...\n(truncated)"


def build_summary_payload(
    case: DebateCaseInput,
    llm_model: Optional[str],
    temperature: float,
    llm_attempts: int = 1,
) -> Dict:
    history_text = summarize_rounds(case.debate_rounds)
    default_summary = {
        "summary": (
            f"Initial mismatch: modelA={case.initial_A_result}, "
            f"modelB={case.initial_B_result}. "
            f"Debate converged in {len(case.debate_rounds)} rounds."
        ),
        "mismatch_reason": "",
        "decisive_argument": "",
        "guardrails": [],
        "modeling_patterns": [],
    }
    if not llm_model:
        return default_summary | {"history_excerpt": history_text}

    prompt = f"""
You are helping an optimisation-debate memory builder.

Problem description:
{case.description}

Initial disagreement:
- Model A result: {case.initial_A_result}
- Model B result: {case.initial_B_result}
- Ground truth (if known): {case.ground_truth}

Debate transcript:
{history_text}

Final consensus objective: {case.final_result}

Please return a JSON object with the following keys:
- "summary": 2-3 sentences explaining how the debate resolved the mismatch.
- "mismatch_reason": concise reason for the disagreement.
- "decisive_argument": specific insight that convinced both sides.
- "guardrails": list of actionable bullet points the next debater should follow.
- "modeling_patterns": list of reusable modeling tricks/structures that appeared.

JSON ONLY. No prose outside the JSON.
""".strip()

    attempts_remaining = max(1, llm_attempts)
    last_error: Optional[Exception] = None
    while attempts_remaining > 0:
        try:
            response = get_response(
                prompt,
                model=llm_model,
                temperature=temperature,
                maximum_retries=1,
            )
            payload = json.loads(response)
            payload["history_excerpt"] = history_text
            return payload
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            attempts_remaining -= 1

    fallback = default_summary.copy()
    failure_reason = f"{last_error}" if last_error else "LLM call failed"
    fallback["summary"] += f" LLM summary failed: {failure_reason}"
    fallback["history_excerpt"] = history_text
    return fallback


def existing_signatures(memory_dir: Path) -> set[str]:
    cases_path = memory_dir / "cases.jsonl"
    if not cases_path.exists():
        return set()
    signs: set[str] = set()
    with cases_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            meta = data.get("metadata") or {}
            sig = meta.get("debate_signature")
            if sig:
                signs.add(sig)
    return signs


class DebateMemoryBuilder:
    def __init__(
        self,
        runs_root: Path,
        memory_dir: Path,
        mismatch_tolerance: float,
        llm_model: Optional[str],
        temperature: float,
        llm_attempts: int,
        max_workers: int,
        datasets: Optional[Iterable[str]] = None,
        dry_run: bool = False,
    ) -> None:
        self.runs_root = runs_root
        self.memory_dir = memory_dir
        self.mismatch_tolerance = mismatch_tolerance
        self.llm_model = llm_model
        self.temperature = temperature
        self.llm_attempts = max(1, llm_attempts)
        self.max_workers = max_workers
        self.datasets_filter = {d.lower() for d in datasets} if datasets else None
        self.dry_run = dry_run

    def build(self) -> None:
        candidates = self._collect_candidates()
        if not candidates:
            print("No qualifying debate cases found.")
            return

        if not self.memory_dir.exists() and not self.dry_run:
            self.memory_dir.mkdir(parents=True, exist_ok=True)

        seen_sigs = existing_signatures(self.memory_dir)

        bank = None if self.dry_run else MemoryBank(memory_dir=str(self.memory_dir))

        added = 0
        skipped_duplicates = 0
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(self._summarize_case, case): case
                for case in candidates
            }
            for future in tqdm(as_completed(futures), total=len(futures), desc="Summarizing"):
                case = futures[future]
                signature = f"{case.dataset}:{case.problem_id}:{case.run_dir.name}"
                if signature in seen_sigs:
                    skipped_duplicates += 1
                    continue
                summary_payload = future.result()
                description = (
                    f"{case.description.strip()}\n\n"
                    f"# Debate Memory Summary\n"
                    f"{summary_payload.get('summary', '').strip()}"
                ).strip()
                metadata = {
                    "source": "debate_memory_builder",
                    "dataset": case.dataset,
                    "run_dir": str(case.run_dir),
                    "modelA": case.modelA,
                    "modelB": case.modelB,
                    "initial_A_result": case.initial_A_result,
                    "initial_B_result": case.initial_B_result,
                    "ground_truth": case.ground_truth,
                    "debate_signature": signature,
                    "summary": summary_payload,
                }
                if self.dry_run:
                    added += 1
                    continue
                try:
                    bank.add_case(
                        problem_id=case.problem_id,
                        problem_desc=description,
                        solution_code=case.final_code,
                        objective_value=case.final_result or 0.0,
                        is_correct=True,
                        metadata=metadata,
                    )
                    added += 1
                    seen_sigs.add(signature)
                except Exception as exc:  # noqa: BLE001
                    print(f"Failed to add case {signature}: {exc}")

        print("===== Debate Memory Builder Summary =====")
        print(f"Runs root:      {self.runs_root}")
        print(f"Output dir:     {self.memory_dir}")
        print(f"Total candidates: {len(candidates)}")
        print(f"Added cases:      {added}")
        print(f"Duplicates skipped: {skipped_duplicates}")
        if self.dry_run:
            print("Dry-run mode: no cases were written.")

    def _collect_candidates(self) -> List[DebateCaseInput]:
        candidates: List[DebateCaseInput] = []
        if not self.runs_root.exists():
            print(f"Runs root not found: {self.runs_root}")
            return candidates

        for dataset_dir in sorted(self.runs_root.iterdir()):
            if not dataset_dir.is_dir():
                continue
            dataset_name = dataset_dir.name
            if self.datasets_filter and dataset_name.lower() not in self.datasets_filter:
                continue
            for run_dir in sorted(dataset_dir.iterdir()):
                if not run_dir.is_dir():
                    continue
                dataset_candidates = self._parse_run(dataset_name, run_dir)
                candidates.extend(dataset_candidates)
        return candidates

    def _parse_run(self, dataset: str, run_dir: Path) -> List[DebateCaseInput]:
        results_path = run_dir / "debate_results.jsonl"
        if not results_path.exists():
            return []

        modelA, modelB = self._infer_models(run_dir.name)
        consensus_path = next(run_dir.glob("consensus_*_vs_*.jsonl"), None)
        consensus_records = load_jsonl(consensus_path) if consensus_path else []
        desc_map = {int(rec["id"]): rec for rec in consensus_records if "id" in rec}

        eval_path = run_dir / "eval_consensus" / "evaluation_results.jsonl"
        evaluation_map = {
            int(rec["id"]): rec for rec in load_jsonl(eval_path) if "id" in rec
        }

        run_candidates: List[DebateCaseInput] = []
        for entry in load_jsonl(results_path):
            problem_id = entry.get("problem_id")
            if problem_id is None:
                continue
            problem_id = int(problem_id)
            if not has_disagreement(entry, self.mismatch_tolerance):
                continue
            if not entry.get("converged"):
                continue
            evaluation = evaluation_map.get(problem_id)
            desc_entry = desc_map.get(problem_id)
            if desc_entry:
                description = desc_entry.get("description") or f"{dataset} problem {problem_id}"
            else:
                description = f"Dataset {dataset} problem {problem_id}"
            final_code = entry.get("final_code") or (
                desc_entry.get("generated_code", "") if desc_entry else ""
            )
            if not final_code:
                continue
            debate_rounds = entry.get("debate_rounds") or []
            if not debate_rounds:
                continue
            run_candidates.append(
                DebateCaseInput(
                    dataset=dataset,
                    problem_id=problem_id,
                    description=description,
                    final_code=final_code,
                    final_result=float_or_none(entry.get("final_result")),
                    debate_rounds=debate_rounds,
                    modelA=modelA,
                    modelB=modelB,
                    run_dir=run_dir,
                    ground_truth=entry.get("ground_truth"),
                    initial_A_result=float_or_none(entry.get("initial_A_result")),
                    initial_B_result=float_or_none(entry.get("initial_B_result")),
                    evaluation=evaluation or {},
                    metadata={
                        "run_dir": str(run_dir),
                        "dataset": dataset,
                    },
                )
            )
        return run_candidates

    @staticmethod
    def _infer_models(run_name: str) -> Tuple[str, str]:
        """
        Run folder format: <timestamp>_<modelA>_vs_<modelB>
        """
        parts = run_name.split("_vs_")
        if len(parts) != 2:
            return "modelA", "modelB"
        left = parts[0].split("_")  # timestamp + modelA pieces
        if len(left) < 2:
            return left[-1], parts[1]
        modelA = "_".join(left[1:])
        modelB = parts[1]
        return modelA, modelB

    def _summarize_case(self, case: DebateCaseInput) -> Dict:
        return build_summary_payload(
            case,
            llm_model=self.llm_model,
            temperature=self.temperature,
            llm_attempts=self.llm_attempts,
        )


def parse_args():
    parser = argparse.ArgumentParser(description="Build debate memory bank from historical runs.")
    parser.add_argument(
        "--runs_root",
        type=str,
        default=str(DEFAULT_RUNS_ROOT),
        help="Directory containing debate run artifacts.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(DEFAULT_DEBATE_MEMORY_DIR),
        help="Directory to store the debate memory bank.",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="*",
        default=None,
        help="Optional dataset filters (case-insensitive).",
    )
    parser.add_argument(
        "--mismatch_tolerance",
        type=float,
        default=1e-3,
        help="Minimum absolute difference between initial results to consider a disagreement.",
    )
    parser.add_argument(
        "--llm_model",
        type=str,
        default=None,
        help="Optional model name for LLM-based summaries. If omitted, heuristic summaries are used.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.3,
        help="Temperature for LLM summaries.",
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=4,
        help="Parallel workers for summary generation.",
    )
    parser.add_argument(
        "--llm_attempts",
        type=int,
        default=2,
        help="Number of LLM attempts per case before falling back to heuristics.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Run the pipeline without writing to the memory bank.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    builder = DebateMemoryBuilder(
        runs_root=Path(args.runs_root),
        memory_dir=Path(args.output_dir),
        mismatch_tolerance=args.mismatch_tolerance,
        llm_model=args.llm_model,
        temperature=args.temperature,
        llm_attempts=args.llm_attempts,
        max_workers=args.max_workers,
        datasets=args.datasets,
        dry_run=args.dry_run,
    )
    builder.build()


if __name__ == "__main__":
    main()
