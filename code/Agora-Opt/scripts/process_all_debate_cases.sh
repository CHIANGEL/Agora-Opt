#!/bin/bash

# Batch process every historical debate run and refresh the debate memory bank.
#
# Usage:
#   ./scripts/process_all_debate_cases.sh [runs_root] [output_dir]
# Example:
#   ./scripts/process_all_debate_cases.sh \
#       ../../results/Agora-Opt/debate \
#       debate_memory_storage
#
# Environment variables (optional):
#   LLM_MODEL        - override default gpt-4o summarizer
#   LLM_ATTEMPTS     - retries per case (default 2)
#   MAX_WORKERS      - thread pool size (default 64)
#   PYTHON_BIN       - python executable (default python)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DEFAULT_RUNS_ROOT="${PROJECT_ROOT}/../../results/Agora-Opt/debate"

RUNS_ROOT="${1:-$DEFAULT_RUNS_ROOT}"
OUTPUT_DIR="${2:-${PROJECT_ROOT}/debate_memory_storage}"

LLM_MODEL="${LLM_MODEL:-gpt-4o}"
LLM_ATTEMPTS="${LLM_ATTEMPTS:-2}"
MAX_WORKERS="${MAX_WORKERS:-64}"
PYTHON_BIN="${PYTHON_BIN:-python}"

echo "============================================================"
echo "🧠 Building Debate Memory"
echo "============================================================"
echo "Runs root:       ${RUNS_ROOT}"
echo "Output dir:      ${OUTPUT_DIR}"
echo "LLM model:       ${LLM_MODEL:-<heuristic>}"
echo "LLM attempts:    ${LLM_ATTEMPTS}"
echo "Max workers:     ${MAX_WORKERS}"
echo "Python binary:   ${PYTHON_BIN}"
echo "============================================================"
echo

CMD=(
  "${PYTHON_BIN}"
  "${PROJECT_ROOT}/scripts/build_debate_memory.py"
  "--runs_root" "${RUNS_ROOT}"
  "--output_dir" "${OUTPUT_DIR}"
  "--max_workers" "${MAX_WORKERS}"
  "--llm_attempts" "${LLM_ATTEMPTS}"
)

if [ -n "${LLM_MODEL}" ]; then
  CMD+=("--llm_model" "${LLM_MODEL}")
fi

echo "Running: ${CMD[*]}"
echo

"${CMD[@]}"

echo
echo "✅ Debate memory refreshed."
echo "Cases stored in: ${OUTPUT_DIR}"
