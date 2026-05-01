#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AGORA_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
OPEN_ROOT="$(cd "${AGORA_DIR}/../.." && pwd)"
RESULTS_ROOT="${OPEN_ROOT}/results"
BENCHMARK_DIR="${OPEN_ROOT}/data/benchmarks"
PYTHON_BIN="${PYTHON_BIN:-python3}"

SOLUTION_MEMORY_DIR="${SOLUTION_MEMORY_DIR:-${AGORA_DIR}/memory_storage}"
DEBUG_CASE_MEMORY_DIR="${DEBUG_CASE_MEMORY_DIR:-${AGORA_DIR}/debug_case_memory}"
DEBATE_MEMORY_DIR="${DEBATE_MEMORY_DIR:-${AGORA_DIR}/debate_memory_storage}"
DEBATE_RUNS_ROOT="${DEBATE_RUNS_ROOT:-${RESULTS_ROOT}/Agora-Opt/debate}"

export PYTHONPATH="${AGORA_DIR}/src:${PYTHONPATH:-}"

echo "============================================================"
echo "Agora-Opt Memory Builder"
echo "============================================================"
echo "Solution memory: ${SOLUTION_MEMORY_DIR}"
echo "Debug memory:    ${DEBUG_CASE_MEMORY_DIR}"
echo "Debate memory:   ${DEBATE_MEMORY_DIR}"
echo "Debate runs:     ${DEBATE_RUNS_ROOT}"
echo "============================================================"
echo

if [[ "$#" -gt 0 ]]; then
    echo "Building solution memory from evaluation directories..."
    "${PYTHON_BIN}" "${SCRIPT_DIR}/build_memory_from_eval_results.py" \
        --eval_dirs "$@" \
        --benchmarks_dir "${BENCHMARK_DIR}" \
        --memory_dir "${SOLUTION_MEMORY_DIR}"
    echo
else
    echo "Skipping solution memory rebuild because no evaluation directories were provided."
    echo "Usage example:"
    echo "  bash ./code/Agora-Opt/scripts/build_memory_assets.sh /path/to/eval_dir1 /path/to/eval_dir2"
    echo
fi

echo "Building debug memory..."
"${PYTHON_BIN}" "${SCRIPT_DIR}/build_debug_memory.py" \
    --output_dir "${DEBUG_CASE_MEMORY_DIR}"
echo

if [[ -d "${DEBATE_RUNS_ROOT}" ]]; then
    echo "Building debate memory..."
    "${PYTHON_BIN}" "${SCRIPT_DIR}/build_debate_memory.py" \
        --runs_root "${DEBATE_RUNS_ROOT}" \
        --output_dir "${DEBATE_MEMORY_DIR}"
else
    echo "Skipping debate memory rebuild because debate runs root does not exist:"
    echo "  ${DEBATE_RUNS_ROOT}"
fi
