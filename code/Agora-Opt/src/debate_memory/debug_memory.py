# -*- coding: utf-8 -*-
"""Lightweight persistence for debugging experiences."""

from __future__ import annotations

import hashlib
import json
import threading
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalise_error(text: str) -> str:
    return (text or "").strip()


@dataclass
class DebugRecord:
    """Single debugging observation stored on disk."""

    signature: str
    status: str
    error_text: str
    guidance: str
    problem_id: Optional[int]
    description: str
    metadata: Dict[str, Any]
    timestamp: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


_PKG_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _PKG_DIR.parent.parent


class DebugMemoryStore:
    """Append-only store keyed by error signature."""

    DEFAULT_PATH = _PROJECT_ROOT / "memory_storage" / "debug_memory.jsonl"

    def __init__(self, path: Optional[str] = None):
        self.path = Path(path) if path else self.DEFAULT_PATH
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.touch()
        self._lock = threading.Lock()

    @staticmethod
    def _signature_from_error(error_text: str, status: str) -> str:
        basis = _normalise_error(error_text)
        if not basis:
            basis = status or "unknown"
        digest = hashlib.sha1(basis.encode("utf-8")).hexdigest()[:12]
        return digest

    def _append(self, record: DebugRecord) -> None:
        payload = json.dumps(record.to_dict(), ensure_ascii=False)
        with self._lock, self.path.open("a", encoding="utf-8") as fh:
            fh.write(payload + "\n")

    def record_execution_feedback(
        self,
        *,
        problem_id: Optional[int],
        description: str,
        status: str,
        error_text: str,
        guidance: str,
        source: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Persist execution feedback and return the signature used."""
        signature_core = self._signature_from_error(error_text, status)
        signature = f"exec:{signature_core}"
        record = DebugRecord(
            signature=signature,
            status=status or "unknown",
            error_text=_normalise_error(error_text) or status or "",
            guidance=(guidance or "").strip(),
            problem_id=problem_id,
            description=(description or "").strip(),
            metadata={
                "source": source,
                **(metadata or {}),
            },
            timestamp=_now_iso(),
        )
        self._append(record)
        return signature

    def record_validation_feedback(
        self,
        *,
        problem_id: Optional[int],
        issues: Iterable[str],
        metadata: Optional[Dict[str, Any]] = None,
        source: str = "validation",
    ) -> List[str]:
        """Persist validation feedback items and return the signatures used."""
        signatures: List[str] = []
        for issue in issues:
            if not issue:
                continue
            signature_core = self._signature_from_error(issue, "validation")
            signature = f"validation:{signature_core}"
            record = DebugRecord(
                signature=signature,
                status="validation",
                error_text=_normalise_error(issue),
                guidance="",
                problem_id=problem_id,
                description="",
                metadata={
                    "source": source,
                    **(metadata or {}),
                },
                timestamp=_now_iso(),
            )
            self._append(record)
            signatures.append(signature)
        return signatures

    def retrieve_for_problem(self, problem_id: int, limit: int = 3) -> List[DebugRecord]:
        """Return recent records for a given problem id (best-effort)."""
        if problem_id is None:
            return []
        matches: List[DebugRecord] = []
        with self.path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if payload.get("problem_id") != problem_id:
                    continue
                matches.append(
                    DebugRecord(
                        signature=payload.get("signature", ""),
                        status=payload.get("status", ""),
                        error_text=payload.get("error_text", ""),
                        guidance=payload.get("guidance", ""),
                        problem_id=payload.get("problem_id"),
                        description=payload.get("description", ""),
                        metadata=payload.get("metadata", {}) or {},
                        timestamp=payload.get("timestamp", ""),
                    )
                )
        matches.sort(key=lambda item: item.timestamp, reverse=True)
        return matches[:limit] if limit else matches


__all__ = ["DebugMemoryStore", "DebugRecord"]
