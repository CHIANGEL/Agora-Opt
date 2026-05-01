# -*- coding: utf-8 -*-
"""Execute generated Python code and capture basic diagnostics."""

from __future__ import annotations

import os
import re
import subprocess
import sys
from dataclasses import dataclass
from typing import Optional


AUTO_OBJECTIVE_SNIPPET = """
# Auto-added snippet: attempt to print the objective value for downstream evaluation.
try:
    candidate = None
    for name in ("model", "m", "Model"):
        if name in globals():
            candidate = globals()[name]
            break
    if candidate is not None and hasattr(candidate, "objVal"):
        print(f"OBJECTIVE_VALUE: {candidate.objVal}")
except Exception:
    pass
""".strip()


@dataclass
class ExecutionResult:
    status: str
    stdout: str
    stderr: str
    objective_value: Optional[float]
    returncode: Optional[int]
    code_path: Optional[str]


def _ensure_directory(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _append_objective_snippet(code: str) -> str:
    if "OBJECTIVE_VALUE" in code:
        return code if code.endswith("\n") else code + "\n"
    return f"{code.rstrip()}\n\n{AUTO_OBJECTIVE_SNIPPET}\n"


def _normalize_output(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _extract_objective_value(output: str) -> Optional[float]:
    if not output:
        return None
    patterns = [
        r"OBJECTIVE_VALUE:\s*([+-]?\d+\.?\d*(?:[eE][+-]?\d+)?)",
        r"Optimal\s+[Oo]bjective[:\s]+([+-]?\d+\.?\d*(?:[eE][+-]?\d+)?)",
        r"Obj:\s*([+-]?\d+\.?\d*(?:[eE][+-]?\d+)?)",
        r"Objective\s+value:\s*([+-]?\d+\.?\d*(?:[eE][+-]?\d+)?)",
    ]
    for pattern in patterns:
        match = re.search(pattern, output, re.IGNORECASE)
        if not match:
            continue
        try:
            return float(match.group(1))
        except ValueError:
            continue
    return None


def execute_generated_code(
    code: str,
    problem_id: int,
    output_dir: str,
    timeout: int = 120,
) -> ExecutionResult:
    """Write code to disk, execute it, and capture the outcome."""
    code_dir = os.path.join(output_dir, "code")
    _ensure_directory(code_dir)

    code_with_snippet = _append_objective_snippet(code)
    code_file = os.path.join(code_dir, f"problem_{problem_id}.py")
    with open(code_file, "w", encoding="utf-8") as fh:
        fh.write(code_with_snippet)

    try:
        completed = subprocess.run(
            [sys.executable, os.path.basename(code_file)],
            cwd=code_dir,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        return ExecutionResult(
            status="timeout",
            stdout=_normalize_output(exc.stdout),
            stderr=f"Execution timeout after {timeout} seconds",
            objective_value=None,
            returncode=None,
            code_path=code_file,
        )
    except Exception as exc:  # pragma: no cover - defensive
        return ExecutionResult(
            status="error",
            stdout="",
            stderr=str(exc),
            objective_value=None,
            returncode=None,
            code_path=code_file,
        )

    stdout = _normalize_output(completed.stdout)
    stderr = _normalize_output(completed.stderr)
    returncode = completed.returncode

    status = "success" if returncode == 0 else "execution_error"
    objective_value = _extract_objective_value(stdout) if status == "success" else None

    return ExecutionResult(
        status=status,
        stdout=stdout,
        stderr=stderr,
        objective_value=objective_value,
        returncode=returncode,
        code_path=code_file,
    )


__all__ = ["ExecutionResult", "execute_generated_code"]
