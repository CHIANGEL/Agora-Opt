# -*- coding: utf-8 -*-
"""Minimal helpers for generated code execution reports."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict
from typing import List, Optional

from .debug_memory import DebugMemoryStore


@dataclass
class DebugMetadata:
    problem_id: int
    notes: List[str]

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, indent=2)


def sanitize_code(code: str, problem_id: int):
    """Ensure code ends with a newline and capture any lightweight notes."""
    metadata = DebugMetadata(problem_id=problem_id, notes=[])
    cleaned = (code or "").rstrip() + "\n" if code else ""
    return cleaned, metadata


def save_debug_metadata(metadata: DebugMetadata, output_dir: str) -> None:
    """Persist metadata only when there is something noteworthy."""
    if not metadata.notes:
        return
    debug_dir = os.path.join(output_dir, "debug")
    os.makedirs(debug_dir, exist_ok=True)
    path = os.path.join(debug_dir, f"problem_{metadata.problem_id}.json")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(metadata.to_json())


def write_debug_suggestions(
    problem_id: int,
    description: str,
    error_message: str,
    memory_helper,
    memory_bank,
    output_dir: str,
    *,
    status: str,
    debug_store: Optional[DebugMemoryStore] = None,
    top_k_cases: int = 3,
) -> None:
    """Write a straightforward debug report and optionally record the memory."""
    _ = memory_helper, memory_bank, top_k_cases  # Unused but kept for interface compatibility.
    debug_dir = os.path.join(output_dir, "debug")
    os.makedirs(debug_dir, exist_ok=True)
    path = os.path.join(debug_dir, f"problem_{problem_id}_suggestions.md")

    lines: List[str] = [
        f"# Debug Report for Problem {problem_id}",
        "",
        f"- **Status:** {status}",
    ]
    if description:
        lines.extend(["", "## Description", description.strip(), ""])
    if error_message:
        lines.extend(
            [
                "## Error Traceback",
                "```",
                error_message.strip(),
                "```",
                "",
            ]
        )
    else:
        lines.extend(["", "## Error Traceback", "_No traceback captured._", ""])

    lines.append("## Notes")
    lines.append("")
    lines.append("Automated debugging is not yet implemented. Review the trace above for hints.")
    lines.append("")

    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))

    if debug_store:
        debug_store.record_execution_feedback(
            problem_id=problem_id,
            description=description,
            status=status,
            error_text=error_message or status,
            guidance="Automated debugging is not yet implemented.",
            source="debug_utils.write_debug_suggestions",
            metadata={},
        )


__all__ = ["DebugMetadata", "sanitize_code", "save_debug_metadata", "write_debug_suggestions"]
