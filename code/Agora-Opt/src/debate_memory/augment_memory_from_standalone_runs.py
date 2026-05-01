#!/usr/bin/env python3
"""Build non-destructive memory variants from standalone pipeline runs."""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from llama_index.core import Document

from .memory_bank import MemoryBank

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
DEFAULT_BASE_ROOT = PROJECT_ROOT
DEFAULT_VARIANTS_ROOT = PROJECT_ROOT / "memory_variants"
DEFAULT_STANDALONE_ROOT = Path("/home/datagen/OR-Debate/standalone_pipeline/runs")

MAIN_MEMORY_DIRNAME = "memory_storage"
DEBUG_CASE_MEMORY_DIRNAME = "debug_case_memory"
DEBATE_MEMORY_DIRNAME = "debate_memory_storage"
DEBUG_MEMORY_FILENAME = "debug_memory.jsonl"

DEBUG_FAILURE_STATUSES = {
    "execution_error",
    "error",
    "timeout",
    "no_code",
    "not_executed",
    "success_no_objective",
    "execution_failed",
}

PROMPT_ARTIFACT_HEADERS = (
    "\n# Retrieved Historical Cases",
    "\n# Debate Memory Insights",
    "\n# Retrieved Debug Guidance",
)


@dataclass
class RunArtifacts:
    source_root: Path
    run_dir: Path
    dataset: str
    model_a: str
    model_b: str
    single_generated: Dict[str, Path]
    debate_results: Optional[Path]
    consensus_jsonl: Optional[Path]
    consensus_eval: Optional[Path]
    manifest_path: Optional[Path]

    @property
    def has_complete_debate(self) -> bool:
        return bool(
            self.debate_results
            and self.consensus_jsonl
            and self.consensus_eval
            and self.debate_results.exists()
            and self.consensus_jsonl.exists()
            and self.consensus_eval.exists()
        )


@dataclass
class ReferenceSolution:
    source: str
    model: str
    code: str
    objective_value: Optional[float]
    chosen_model: Optional[str]


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def now_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def load_jsonl(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    if not path or not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def append_jsonl(path: Path, rows: Iterable[Dict]) -> int:
    count = 0
    with path.open("a", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def load_json(path: Path) -> Dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def dump_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2, sort_keys=True)


def count_jsonl_lines(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as fh:
        return sum(1 for _ in fh if _.strip())


def float_or_none(value) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def infer_models_from_run_name(run_name: str) -> Tuple[str, str]:
    parts = run_name.split("_vs_")
    if len(parts) != 2:
        return "modelA", "modelB"
    left = parts[0].split("_")
    if len(left) < 2:
        return left[-1], parts[1]
    return "_".join(left[1:]), parts[1]


def clean_description(text: str) -> str:
    cleaned = (text or "").strip()
    for header in PROMPT_ARTIFACT_HEADERS:
        pos = cleaned.find(header)
        if pos != -1:
            cleaned = cleaned[:pos].rstrip()
    return cleaned


def check_correctness(
    pred_obj: Optional[float],
    gt_obj: Optional[float],
    tolerance: float,
    use_relative_tolerance: bool,
) -> bool:
    if pred_obj is None or gt_obj is None:
        return False
    if gt_obj == 0:
        return abs(pred_obj) <= tolerance
    if use_relative_tolerance:
        return abs((pred_obj - gt_obj) / gt_obj) <= tolerance
    return abs(pred_obj - gt_obj) <= tolerance


def sha1_short(text: str, length: int = 16) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:length]


def build_doc(problem_id: int, description: str, solution_code: str, objective_value: float, metadata: Dict) -> Document:
    doc_text = f"""Problem: {description}

Solution approach:
{solution_code[:500]}...

Key features:
- Problem ID: {problem_id}
- Objective value: {objective_value}
- Status: Correct
"""
    return Document(
        text=doc_text,
        metadata={
            "problem_id": problem_id,
            "objective_value": objective_value,
            **metadata,
        },
    )


class BatchMemoryAppender:
    def __init__(self, memory_dir: Path, embedding_model: str) -> None:
        self.memory_dir = memory_dir
        self.bank = MemoryBank(memory_dir=str(memory_dir), embedding_model=embedding_model)
        self.pending_cases: List[Dict] = []
        self.pending_docs: List[Document] = []

    def add_case(
        self,
        *,
        problem_id: int,
        problem_desc: str,
        solution_code: str,
        objective_value: float,
        metadata: Dict,
    ) -> None:
        case = {
            "problem_id": int(problem_id),
            "description": problem_desc,
            "solution_code": solution_code,
            "objective_value": objective_value,
            "is_correct": True,
            "metadata": metadata,
        }
        self.pending_cases.append(case)
        self.pending_docs.append(
            build_doc(
                problem_id=int(problem_id),
                description=problem_desc,
                solution_code=solution_code,
                objective_value=objective_value,
                metadata=metadata,
            )
        )

    def finalize(self) -> int:
        if not self.pending_cases:
            return 0
        with Path(self.bank.cases_file).open("a", encoding="utf-8") as fh:
            for case in self.pending_cases:
                fh.write(json.dumps(case, ensure_ascii=False) + "\n")
        for doc in self.pending_docs:
            self.bank.index.insert(doc)
        self.bank.index.storage_context.persist(persist_dir=self.bank.index_dir)
        added = len(self.pending_cases)
        self.pending_cases.clear()
        self.pending_docs.clear()
        return added


def resolve_source_roots(patterns: Sequence[str]) -> List[Path]:
    resolved: List[Path] = []
    for pattern in patterns:
        matches = glob.glob(pattern)
        if matches:
            for match in matches:
                path = Path(match)
                if path.is_dir():
                    resolved.append(path.resolve())
        else:
            path = Path(pattern)
            if path.is_dir():
                resolved.append(path.resolve())
    deduped = sorted({path for path in resolved})
    return deduped


def resolve_file(run_dir: Path, raw_value: Optional[str]) -> Optional[Path]:
    if not raw_value:
        return None
    candidate = Path(raw_value)
    if not candidate.is_absolute():
        candidate = run_dir / candidate
    return candidate if candidate.exists() else None


def discover_run_artifacts(source_root: Path) -> List[RunArtifacts]:
    runs: List[RunArtifacts] = []
    if not source_root.exists():
        return runs

    for run_dir in sorted(source_root.iterdir()):
        if not run_dir.is_dir():
            continue

        manifest_path = run_dir / "run_manifest.json"
        manifest = load_json(manifest_path) if manifest_path.exists() else {}

        model_a, model_b = infer_models_from_run_name(run_dir.name)
        model_a = manifest.get("model_a", model_a)
        model_b = manifest.get("model_b", model_b)
        dataset = manifest.get("dataset", source_root.name)

        single_generated: Dict[str, Path] = {}
        for generated in sorted(run_dir.glob("single/*/generated.jsonl")):
            model_name = generated.parent.name
            single_generated[model_name] = generated

        model_a_generated = resolve_file(run_dir, manifest.get("model_a_generated"))
        model_b_generated = resolve_file(run_dir, manifest.get("model_b_generated"))
        if model_a_generated:
            single_generated.setdefault(model_a, model_a_generated)
        if model_b_generated:
            single_generated.setdefault(model_b, model_b_generated)

        debate_results = run_dir / "debate" / "debate_results.jsonl"
        if not debate_results.exists():
            debate_results = resolve_file(run_dir, manifest.get("debate_dir"))
            if debate_results and debate_results.is_dir():
                debate_results = debate_results / "debate_results.jsonl"
        if debate_results and not debate_results.exists():
            debate_results = None

        consensus_jsonl = resolve_file(run_dir, manifest.get("consensus_jsonl"))
        if consensus_jsonl is None:
            candidates = sorted((run_dir / "debate").glob("consensus_*.jsonl"))
            consensus_jsonl = candidates[0] if candidates else None

        consensus_eval = run_dir / "consensus_eval" / "evaluation_results.jsonl"
        if not consensus_eval.exists():
            consensus_eval = None

        runs.append(
            RunArtifacts(
                source_root=source_root,
                run_dir=run_dir,
                dataset=dataset,
                model_a=model_a,
                model_b=model_b,
                single_generated=single_generated,
                debate_results=debate_results,
                consensus_jsonl=consensus_jsonl,
                consensus_eval=consensus_eval,
                manifest_path=manifest_path if manifest_path.exists() else None,
            )
        )
    return runs


def load_existing_case_signatures(cases_file: Path) -> set[str]:
    signatures: set[str] = set()
    if not cases_file.exists():
        return signatures
    with cases_file.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            meta = row.get("metadata") or {}
            for key in ("import_signature", "debate_signature"):
                value = meta.get(key)
                if value:
                    signatures.add(str(value))
    return signatures


def load_existing_debug_signatures(debug_memory_file: Path) -> set[str]:
    signatures: set[str] = set()
    if not debug_memory_file.exists():
        return signatures
    with debug_memory_file.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            signature = row.get("signature")
            if signature:
                signatures.add(str(signature))
    return signatures


def summarize_rounds(rounds: List[Dict], max_chars: int = 1800) -> str:
    if not rounds:
        return ""
    lines: List[str] = []
    for rnd in rounds:
        lines.append(
            f"Round {rnd.get('round')}: "
            f"A={rnd.get('result_A')} ({rnd.get('status_A')}), "
            f"B={rnd.get('result_B')} ({rnd.get('status_B')})"
        )
        analysis_a = (rnd.get("analysis_A") or "").strip()
        analysis_b = (rnd.get("analysis_B") or "").strip()
        if analysis_a:
            lines.append(f"Model A analysis:\n{analysis_a}")
        if analysis_b:
            lines.append(f"Model B analysis:\n{analysis_b}")
        lines.append("")
    text = "\n".join(lines).strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 16] + "\n...\n(truncated)"


def heuristic_debate_summary(entry: Dict, model_a: str, model_b: str) -> Dict:
    initial_a = entry.get("initial_A_result")
    initial_b = entry.get("initial_B_result")
    final_result = entry.get("final_result")
    chosen_model = entry.get("chosen_model") or "consensus"
    rounds = entry.get("debate_rounds") or []
    summary = (
        f"Initial mismatch: {model_a}={initial_a}, {model_b}={initial_b}. "
        f"Debate converged in {len(rounds)} rounds and selected {chosen_model} "
        f"with final objective {final_result}."
    )
    decisive_argument = (
        f"The final candidate from {chosen_model} was retained after both sides "
        "aligned on the same executable outcome."
    )
    guardrails = [
        "Compare feasibility and objective values before rewriting the model.",
        "Keep a stable executable candidate whenever later edits do not improve the result.",
    ]
    return {
        "summary": summary,
        "mismatch_reason": "The two models initially disagreed on the objective value or feasibility.",
        "decisive_argument": decisive_argument,
        "guardrails": guardrails,
        "modeling_patterns": [],
        "history_excerpt": summarize_rounds(rounds),
    }


def guidance_for_status(status: str) -> str:
    status = (status or "").strip()
    if status == "no_code":
        return "Return a complete executable Python program inside a ```python``` block."
    if status == "success_no_objective":
        return "Print the optimized objective explicitly, for example with OBJECTIVE_VALUE after optimize()."
    if status == "timeout":
        return "Reduce model-construction overhead and check whether loops or constraints are exploding combinatorially."
    if status == "not_executed":
        return "Make sure the generated response contains runnable code and that the execution step is actually triggered."
    return "Check imports, indexing, variable names, and model-object references against the traceback."


def has_disagreement(initial_a: Optional[float], initial_b: Optional[float], tolerance: float) -> bool:
    if initial_a is None or initial_b is None:
        return True
    return abs(initial_a - initial_b) > tolerance


def choose_error_text(row: Dict) -> str:
    stderr = (row.get("execution_stderr") or "").strip()
    stdout = (row.get("execution_stdout") or "").strip()
    status = (row.get("execution_status") or row.get("status") or "").strip()
    if stderr:
        return stderr
    if stdout:
        return stdout
    if status == "no_code":
        return "Generated code block is empty."
    if status == "not_executed":
        return "Execution did not complete and no detailed stderr/stdout was recorded."
    if status == "success_no_objective":
        return "Execution succeeded but no objective value could be extracted from stdout."
    return status or "Unknown execution issue."


def clone_base_memory_dirs(base_root: Path, variant_dir: Path) -> Dict[str, Path]:
    mapping = {}
    for dirname in (MAIN_MEMORY_DIRNAME, DEBUG_CASE_MEMORY_DIRNAME, DEBATE_MEMORY_DIRNAME):
        src = base_root / dirname
        dst = variant_dir / dirname
        shutil.copytree(src, dst)
        mapping[dirname] = dst
    return mapping


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create augmented memory-bank variants from standalone pipeline runs without touching originals."
    )
    parser.add_argument(
        "--variant_name",
        type=str,
        required=True,
        help="Name of the output variant directory under memory_variants/",
    )
    parser.add_argument(
        "--source",
        nargs="+",
        required=True,
        help="Source directories or glob patterns under standalone_pipeline/runs.",
    )
    parser.add_argument(
        "--base_root",
        type=str,
        default=str(DEFAULT_BASE_ROOT),
        help="Project root that contains memory_storage/debug_case_memory/debate_memory_storage.",
    )
    parser.add_argument(
        "--variants_root",
        type=str,
        default=str(DEFAULT_VARIANTS_ROOT),
        help="Directory under which new variants are created.",
    )
    parser.add_argument(
        "--embedding_model",
        type=str,
        default="BAAI/bge-small-en-v1.5",
        help="Embedding model name or local path used when updating vector indexes.",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.05,
        help="Correctness tolerance for imported single-model cases.",
    )
    parser.add_argument(
        "--mismatch_tolerance",
        type=float,
        default=1e-3,
        help="Minimum difference between initial debate results to count as a disagreement.",
    )
    parser.add_argument(
        "--use_relative_tolerance",
        action="store_true",
        help="Use relative tolerance when judging single-model correctness.",
    )
    args = parser.parse_args()

    base_root = Path(args.base_root).resolve()
    variants_root = Path(args.variants_root).resolve()
    source_roots = resolve_source_roots(args.source)
    if not source_roots:
        raise FileNotFoundError(f"No source roots matched: {args.source}")

    variant_dir = variants_root / args.variant_name
    if variant_dir.exists():
        raise FileExistsError(f"Variant already exists: {variant_dir}")
    variant_dir.parent.mkdir(parents=True, exist_ok=True)

    print("=== Augment Standalone Memory Banks ===")
    print(f"Base root:      {base_root}")
    print(f"Variant dir:    {variant_dir}")
    print(f"Source roots:   {len(source_roots)}")
    for root in source_roots:
        print(f"  - {root}")

    memory_dirs = clone_base_memory_dirs(base_root, variant_dir)

    main_memory_dir = memory_dirs[MAIN_MEMORY_DIRNAME]
    debug_case_memory_dir = memory_dirs[DEBUG_CASE_MEMORY_DIRNAME]
    debate_memory_dir = memory_dirs[DEBATE_MEMORY_DIRNAME]
    debug_memory_file = main_memory_dir / DEBUG_MEMORY_FILENAME

    main_seen = load_existing_case_signatures(main_memory_dir / "cases.jsonl")
    debug_case_seen = load_existing_case_signatures(debug_case_memory_dir / "cases.jsonl")
    debate_seen = load_existing_case_signatures(debate_memory_dir / "cases.jsonl")
    debug_raw_seen = load_existing_debug_signatures(debug_memory_file)

    main_appender = BatchMemoryAppender(main_memory_dir, args.embedding_model)
    debug_case_appender = BatchMemoryAppender(debug_case_memory_dir, args.embedding_model)
    debate_appender = BatchMemoryAppender(debate_memory_dir, args.embedding_model)
    pending_debug_rows: List[Dict] = []

    stats = {
        "runs": {
            "source_roots": len(source_roots),
            "runs_discovered": 0,
            "runs_with_manifest": 0,
            "runs_with_complete_debate": 0,
            "runs_partial_or_single_only": 0,
        },
        "memory_storage": {
            "single_correct_added": 0,
            "consensus_correct_added": 0,
            "duplicates_skipped": 0,
            "incorrect_or_missing_single_skipped": 0,
            "consensus_missing_code_or_eval_skipped": 0,
        },
        "debug_memory": {
            "raw_records_added": 0,
            "case_records_added": 0,
            "duplicates_skipped": 0,
            "non_failure_skipped": 0,
            "missing_reference_skipped": 0,
        },
        "debate_memory": {
            "added": 0,
            "duplicates_skipped": 0,
            "missing_or_incorrect_skipped": 0,
        },
    }

    all_runs: List[RunArtifacts] = []
    for source_root in source_roots:
        all_runs.extend(discover_run_artifacts(source_root))

    stats["runs"]["runs_discovered"] = len(all_runs)
    stats["runs"]["runs_with_manifest"] = sum(1 for run in all_runs if run.manifest_path)
    stats["runs"]["runs_with_complete_debate"] = sum(1 for run in all_runs if run.has_complete_debate)
    stats["runs"]["runs_partial_or_single_only"] = stats["runs"]["runs_discovered"] - stats["runs"]["runs_with_complete_debate"]

    for run in all_runs:
        print(f"Processing run: {run.run_dir}")

        single_rows_by_model: Dict[str, Dict[int, Dict]] = {}
        correct_single_refs: Dict[int, Dict[str, ReferenceSolution]] = {}

        for model_name, generated_path in sorted(run.single_generated.items()):
            rows_map: Dict[int, Dict] = {}
            for row in load_jsonl(generated_path):
                problem_id = row.get("id")
                if problem_id is None:
                    continue
                try:
                    problem_id = int(problem_id)
                except (TypeError, ValueError):
                    continue
                rows_map[problem_id] = row

                code = (row.get("generated_code") or "").strip()
                pred = float_or_none(row.get("execution_objective_value"))
                gt = float_or_none(row.get("answer"))
                is_correct = bool(code) and check_correctness(
                    pred,
                    gt,
                    tolerance=args.tolerance,
                    use_relative_tolerance=args.use_relative_tolerance,
                )
                if not is_correct:
                    stats["memory_storage"]["incorrect_or_missing_single_skipped"] += 1
                    continue

                description = clean_description(row.get("description", ""))
                signature_basis = (
                    f"main|single|{run.dataset}|{problem_id}|{model_name}|"
                    f"{sha1_short(code, 20)}|{pred}"
                )
                import_signature = f"standalone-main:{sha1_short(signature_basis, 20)}"
                if import_signature in main_seen:
                    stats["memory_storage"]["duplicates_skipped"] += 1
                    continue

                metadata = {
                    "source": "standalone_single_generated",
                    "dataset": run.dataset,
                    "run_dir": str(run.run_dir),
                    "run_name": run.run_dir.name,
                    "source_root": str(run.source_root),
                    "model": model_name,
                    "execution_status": row.get("execution_status", "unknown"),
                    "ground_truth": row.get("answer"),
                    "case_kind": "single",
                    "import_signature": import_signature,
                }
                main_appender.add_case(
                    problem_id=problem_id,
                    problem_desc=description,
                    solution_code=code,
                    objective_value=pred if pred is not None else 0.0,
                    metadata=metadata,
                )
                main_seen.add(import_signature)
                stats["memory_storage"]["single_correct_added"] += 1
                correct_single_refs.setdefault(problem_id, {})[model_name] = ReferenceSolution(
                    source="single",
                    model=model_name,
                    code=code,
                    objective_value=pred,
                    chosen_model=model_name,
                )

            single_rows_by_model[model_name] = rows_map

        consensus_rows_by_id: Dict[int, Dict] = {}
        debate_rows_by_id: Dict[int, Dict] = {}
        eval_rows_by_id: Dict[int, Dict] = {}
        consensus_refs: Dict[int, ReferenceSolution] = {}

        if run.has_complete_debate:
            for row in load_jsonl(run.consensus_jsonl):
                problem_id = row.get("id")
                if problem_id is None:
                    continue
                try:
                    consensus_rows_by_id[int(problem_id)] = row
                except (TypeError, ValueError):
                    continue
            for row in load_jsonl(run.debate_results):
                problem_id = row.get("problem_id")
                if problem_id is None:
                    continue
                try:
                    debate_rows_by_id[int(problem_id)] = row
                except (TypeError, ValueError):
                    continue
            for row in load_jsonl(run.consensus_eval):
                problem_id = row.get("id")
                if problem_id is None:
                    continue
                try:
                    eval_rows_by_id[int(problem_id)] = row
                except (TypeError, ValueError):
                    continue

            for problem_id, eval_row in eval_rows_by_id.items():
                if not eval_row.get("is_correct", False):
                    stats["memory_storage"]["consensus_missing_code_or_eval_skipped"] += 1
                    continue

                consensus_row = consensus_rows_by_id.get(problem_id, {})
                debate_row = debate_rows_by_id.get(problem_id, {})
                code = (consensus_row.get("generated_code") or debate_row.get("final_code") or "").strip()
                if not code:
                    stats["memory_storage"]["consensus_missing_code_or_eval_skipped"] += 1
                    continue

                description = clean_description(
                    consensus_row.get("description")
                    or next(
                        (
                            model_rows[problem_id].get("description")
                            for model_rows in single_rows_by_model.values()
                            if problem_id in model_rows
                        ),
                        f"{run.dataset} problem {problem_id}",
                    )
                )
                pred = float_or_none(eval_row.get("predicted_objective"))
                signature_basis = (
                    f"main|consensus|{run.dataset}|{problem_id}|"
                    f"{sha1_short(code, 20)}|{pred}"
                )
                import_signature = f"standalone-main:{sha1_short(signature_basis, 20)}"
                if import_signature in main_seen:
                    stats["memory_storage"]["duplicates_skipped"] += 1
                else:
                    metadata = {
                        "source": "standalone_consensus_eval",
                        "dataset": run.dataset,
                        "run_dir": str(run.run_dir),
                        "run_name": run.run_dir.name,
                        "source_root": str(run.source_root),
                        "modelA": run.model_a,
                        "modelB": run.model_b,
                        "chosen_model": debate_row.get("chosen_model") or consensus_row.get("chosen_model"),
                        "execution_status": eval_row.get("execution_status", "unknown"),
                        "ground_truth": eval_row.get("ground_truth"),
                        "case_kind": "consensus",
                        "import_signature": import_signature,
                    }
                    main_appender.add_case(
                        problem_id=problem_id,
                        problem_desc=description,
                        solution_code=code,
                        objective_value=pred if pred is not None else 0.0,
                        metadata=metadata,
                    )
                    main_seen.add(import_signature)
                    stats["memory_storage"]["consensus_correct_added"] += 1

                consensus_refs[problem_id] = ReferenceSolution(
                    source="consensus",
                    model="debate_consensus",
                    code=code,
                    objective_value=pred,
                    chosen_model=debate_row.get("chosen_model") or consensus_row.get("chosen_model"),
                )

            for problem_id, debate_row in debate_rows_by_id.items():
                eval_row = eval_rows_by_id.get(problem_id)
                if not eval_row or not eval_row.get("is_correct", False):
                    stats["debate_memory"]["missing_or_incorrect_skipped"] += 1
                    continue
                if not debate_row.get("converged"):
                    stats["debate_memory"]["missing_or_incorrect_skipped"] += 1
                    continue
                initial_a = float_or_none(debate_row.get("initial_A_result"))
                initial_b = float_or_none(debate_row.get("initial_B_result"))
                if not has_disagreement(initial_a, initial_b, args.mismatch_tolerance):
                    stats["debate_memory"]["missing_or_incorrect_skipped"] += 1
                    continue

                final_code = (debate_row.get("final_code") or "").strip()
                if not final_code:
                    stats["debate_memory"]["missing_or_incorrect_skipped"] += 1
                    continue

                base_desc = clean_description(
                    consensus_rows_by_id.get(problem_id, {}).get("description")
                    or next(
                        (
                            model_rows[problem_id].get("description")
                            for model_rows in single_rows_by_model.values()
                            if problem_id in model_rows
                        ),
                        f"{run.dataset} problem {problem_id}",
                    )
                )
                summary_payload = heuristic_debate_summary(debate_row, run.model_a, run.model_b)
                full_desc = (
                    f"{base_desc}\n\n# Debate Memory Summary\n"
                    f"{summary_payload.get('summary', '').strip()}"
                ).strip()
                debate_signature = (
                    f"standalone-debate:{run.dataset}:{problem_id}:{sha1_short(final_code, 20)}"
                )
                if debate_signature in debate_seen:
                    stats["debate_memory"]["duplicates_skipped"] += 1
                    continue

                metadata = {
                    "source": "standalone_debate_memory_import",
                    "dataset": run.dataset,
                    "run_dir": str(run.run_dir),
                    "run_name": run.run_dir.name,
                    "source_root": str(run.source_root),
                    "modelA": run.model_a,
                    "modelB": run.model_b,
                    "initial_A_result": initial_a,
                    "initial_B_result": initial_b,
                    "ground_truth": eval_row.get("ground_truth"),
                    "debate_signature": debate_signature,
                    "import_signature": debate_signature,
                    "summary": summary_payload,
                }
                debate_appender.add_case(
                    problem_id=problem_id,
                    problem_desc=full_desc,
                    solution_code=final_code,
                    objective_value=float_or_none(debate_row.get("final_result")) or 0.0,
                    metadata=metadata,
                )
                debate_seen.add(debate_signature)
                stats["debate_memory"]["added"] += 1

        for model_name, rows_map in sorted(single_rows_by_model.items()):
            for problem_id, row in rows_map.items():
                status = row.get("execution_status") or row.get("status") or ""
                if status not in DEBUG_FAILURE_STATUSES:
                    stats["debug_memory"]["non_failure_skipped"] += 1
                    continue

                reference: Optional[ReferenceSolution] = None
                for other_model, ref in sorted(correct_single_refs.get(problem_id, {}).items()):
                    if other_model != model_name:
                        reference = ref
                        break
                if reference is None:
                    reference = consensus_refs.get(problem_id)
                if reference is None:
                    stats["debug_memory"]["missing_reference_skipped"] += 1
                    continue

                description = clean_description(row.get("description", ""))
                error_text = choose_error_text(row)
                guidance = (
                    f"{guidance_for_status(status)} "
                    f"Reference fix source: {reference.source} ({reference.model}); "
                    f"target objective: {reference.objective_value}."
                )
                import_signature = (
                    f"standalone-debug:{sha1_short(f'{run.dataset}|{problem_id}|{model_name}|{status}|{error_text}|{sha1_short(reference.code, 16)}', 20)}"
                )
                if import_signature in debug_case_seen or import_signature in debug_raw_seen:
                    stats["debug_memory"]["duplicates_skipped"] += 1
                    continue

                debug_record = {
                    "signature": import_signature,
                    "status": status,
                    "error_text": error_text,
                    "guidance": guidance,
                    "problem_id": problem_id,
                    "description": description,
                    "metadata": {
                        "source": "standalone_runs.synthetic_debug_case",
                        "dataset": run.dataset,
                        "run_dir": str(run.run_dir),
                        "run_name": run.run_dir.name,
                        "source_root": str(run.source_root),
                        "model": model_name,
                        "reference_source": reference.source,
                        "reference_model": reference.model,
                        "reference_objective": reference.objective_value,
                        "reference_chosen_model": reference.chosen_model,
                    },
                    "timestamp": now_iso(),
                }
                pending_debug_rows.append(debug_record)
                debug_raw_seen.add(import_signature)

                prompt_desc = (
                    f"{description}\n\n"
                    f"## Error Details\n```\n{error_text}\n```\n"
                    f"## Guidance\n{guidance}\n"
                )
                reference_code = reference.code.strip()
                solution_code = (
                    "# Synthetic Debug Memory Case\n"
                    f"# Signature: {import_signature}\n"
                    f"# Status: {status}\n"
                    f"# Reference source: {reference.source} ({reference.model})\n\n"
                    f"{reference_code}"
                )
                metadata = {
                    "source": "standalone_runs.synthetic_debug_case",
                    "dataset": run.dataset,
                    "run_dir": str(run.run_dir),
                    "run_name": run.run_dir.name,
                    "source_root": str(run.source_root),
                    "model": model_name,
                    "status": status,
                    "signature": import_signature,
                    "reference_source": reference.source,
                    "reference_model": reference.model,
                    "reference_objective": reference.objective_value,
                    "reference_chosen_model": reference.chosen_model,
                    "import_signature": import_signature,
                }
                debug_case_appender.add_case(
                    problem_id=problem_id,
                    problem_desc=prompt_desc,
                    solution_code=solution_code,
                    objective_value=0.0,
                    metadata=metadata,
                )
                debug_case_seen.add(import_signature)
                stats["debug_memory"]["raw_records_added"] += 1
                stats["debug_memory"]["case_records_added"] += 1

    append_jsonl(debug_memory_file, pending_debug_rows)

    main_added = main_appender.finalize()
    debug_case_added = debug_case_appender.finalize()
    debate_added = debate_appender.finalize()

    summary = {
        "created_at": now_iso(),
        "variant_dir": str(variant_dir),
        "base_root": str(base_root),
        "source_patterns": list(args.source),
        "resolved_source_roots": [str(path) for path in source_roots],
        "embedding_model": args.embedding_model,
        "tolerance": args.tolerance,
        "use_relative_tolerance": args.use_relative_tolerance,
        "mismatch_tolerance": args.mismatch_tolerance,
        "stats": stats,
        "final_counts": {
            "memory_storage_cases": count_jsonl_lines(main_memory_dir / "cases.jsonl"),
            "debug_memory_records": count_jsonl_lines(debug_memory_file),
            "debug_case_memory_cases": count_jsonl_lines(debug_case_memory_dir / "cases.jsonl"),
            "debate_memory_cases": count_jsonl_lines(debate_memory_dir / "cases.jsonl"),
            "main_added_persisted": main_added,
            "debug_case_added_persisted": debug_case_added,
            "debate_added_persisted": debate_added,
        },
    }
    dump_json(variant_dir / "import_summary.json", summary)

    print("=== Import Complete ===")
    print(f"Variant:                 {variant_dir}")
    print(f"Main memory added:       {main_added}")
    print(f"Debug raw added:         {len(pending_debug_rows)}")
    print(f"Debug case added:        {debug_case_added}")
    print(f"Debate memory added:     {debate_added}")
    print(f"Summary:                 {variant_dir / 'import_summary.json'}")


if __name__ == "__main__":
    main()
