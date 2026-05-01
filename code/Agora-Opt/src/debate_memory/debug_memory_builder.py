"""Convert debug_memory.jsonl records into a searchable MemoryBank."""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from .memory_bank import MemoryBank

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LEGACY_ROOT = PROJECT_ROOT.parent / "debate_with_memory"


def _default_inputs() -> List[str]:
    candidates = [
        PROJECT_ROOT / "memory_storage" / "debug_memory.jsonl",
        LEGACY_ROOT / "memory_storage" / "debug_memory.jsonl",
        PROJECT_ROOT / "memory_storage" / "backups" / "*" / "debug_memory.jsonl",
        LEGACY_ROOT / "memory_storage" / "backups" / "*" / "debug_memory.jsonl",
    ]
    return [str(path) for path in candidates]


def _stable_id(signature: str) -> int:
    digest = hashlib.sha1(signature.encode("utf-8")).hexdigest()
    return int(digest[:12], 16)


def _parse_timestamp(ts: Optional[str]) -> datetime:
    if not ts:
        return datetime.min
    try:
        return datetime.fromisoformat(ts)
    except ValueError:
        return datetime.min


def load_debug_records(input_globs: List[str]) -> Dict[str, Dict]:
    records: Dict[str, Dict] = {}
    files: List[str] = []
    for pattern in input_globs:
        files.extend(glob.glob(pattern))
    files = sorted({Path(f) for f in files if Path(f).exists()})
    for file_path in files:
        with file_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                signature = record.get("signature")
                if not signature:
                    continue
                ts = _parse_timestamp(record.get("timestamp"))
                existing = records.get(signature)
                if existing is None or ts > existing.get("_ts", datetime.min):
                    record["_ts"] = ts
                    records[signature] = record
    return records


def build_debug_memory(records: Dict[str, Dict], output_dir: Path, clear: bool) -> None:
    if clear and output_dir.exists():
        for child in output_dir.iterdir():
            if child.is_file():
                child.unlink()
            else:
                import shutil

                shutil.rmtree(child)
    bank = MemoryBank(memory_dir=str(output_dir))
    added = 0
    for signature, record in records.items():
        description = record.get("description", "Unknown problem")
        error_text = record.get("error_text", "")
        guidance = record.get("guidance", "")
        status = record.get("status", "")
        metadata = {
            "signature": signature,
            "status": status,
            "timestamp": record.get("timestamp"),
            **(record.get("metadata") or {}),
        }
        note_lines = ["# Debug Memory Case", f"Signature: {signature}", f"Status: {status}"]
        if guidance:
            note_lines.append(f"Guidance: {guidance}")
        note_lines.append("---")
        if error_text:
            note_lines.append("Error snippet:\n" + error_text)
        note_lines.append("---")
        note_lines.append(f"Source metadata: {metadata}")
        prompt_desc = (
            f"{description}\n\n## Error Details\n```\n{error_text}\n```\n"
            f"## Guidance\n{guidance or 'N/A'}\n"
        )
        problem_id = record.get("problem_id")
        if problem_id is None:
            problem_id = _stable_id(signature)
        try:
            bank.add_case(
                problem_id=int(problem_id),
                problem_desc=prompt_desc,
                solution_code="\n".join(note_lines),
                objective_value=0.0,
                is_correct=True,
                metadata=metadata,
            )
            added += 1
        except Exception as exc:  # noqa: BLE001
            print(f"Failed to add debug case {signature}: {exc}")
    print(f"✅ Added {added} debug cases to {output_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description="Build debug memory bank from debug_memory.jsonl records")
    parser.add_argument(
        "--input", nargs="*", default=_default_inputs(), help="Input files/globs containing debug records",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(PROJECT_ROOT / "debug_case_memory"),
        help="Where to store the constructed memory bank",
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Remove existing output_dir contents before rebuilding",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    records = load_debug_records(args.input)
    print(f"Loaded {len(records)} unique debug signatures")
    build_debug_memory(records, Path(args.output_dir), clear=args.clear)


if __name__ == "__main__":
    main()

