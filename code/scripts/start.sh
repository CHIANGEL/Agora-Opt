#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
AGORA_DIR="${PROJECT_ROOT}/code/Agora-Opt"
DATA_DIR="${PROJECT_ROOT}/data/benchmarks"
RESULTS_DIR="${PROJECT_ROOT}/results/Agora-Opt-QuickStart"
GENERATION_DIR="${RESULTS_DIR}/generation"
DEBATE_DIR="${RESULTS_DIR}/debate"
LOG_DIR="${RESULTS_DIR}/logs"
PYTHON_BIN="${PYTHON_BIN:-python3}"
RUN_ID="${RUN_ID:-$(date -u +"%Y%m%d_%H%M%S")}"

# API credential template.
: "${LLM_API_KEY:=PUT YOUR API KEY HERE}"
: "${LLM_API_BASE_URL:=PUT YOUR API URL HERE}"
export LLM_API_KEY
export LLM_API_BASE_URL

# Compatibility mirrors.
: "${OPENAI_API_KEY:=${LLM_API_KEY}}"
: "${OPENAI_BASE_URL:=${LLM_API_BASE_URL}}"
: "${API_KEY:=${LLM_API_KEY}}"
: "${API_URL:=${LLM_API_BASE_URL}}"
export OPENAI_API_KEY
export OPENAI_BASE_URL
export API_KEY
export API_URL

if [[ "${LLM_API_KEY}" == "PUT YOUR API KEY HERE" || "${LLM_API_BASE_URL}" == "PUT YOUR API URL HERE" ]]; then
    echo "Please set LLM_API_KEY and LLM_API_BASE_URL before running this script."
    exit 1
fi

MODEL_A="${MODEL_A:-gpt-4o}"
MODEL_B="${MODEL_B:-deepseek-chat}"
DATASET="${1:-${DATASET:-EasyLP}}"
MAX_PROBLEMS="${2:-${MAX_PROBLEMS:-5}}"
TEMPERATURE="${TEMPERATURE:-0.01}"
GEN_PARALLEL="${GEN_PARALLEL:-4}"
DEBATE_WORKERS="${DEBATE_WORKERS:-4}"
EVAL_WORKERS="${EVAL_WORKERS:-8}"
MAX_ROUNDS="${MAX_ROUNDS:-2}"
TIMEOUT="${TIMEOUT:-90}"
TOLERANCE="${TOLERANCE:-0.05}"

export PYTHONPATH="${AGORA_DIR}/src:${PYTHONPATH:-}"

mkdir -p "${GENERATION_DIR}" "${DEBATE_DIR}" "${LOG_DIR}"

require_path() {
    local path="$1"
    if [[ ! -e "${path}" ]]; then
        echo "Required path not found: ${path}"
        exit 1
    fi
}

require_path "${AGORA_DIR}/scripts/generate_with_memory.py"
require_path "${AGORA_DIR}/scripts/run_memory_debate.py"
require_path "${DATA_DIR}/${DATASET}.jsonl"

run_generation() {
    local model="$1"
    local output_file="${GENERATION_DIR}/${model}_${DATASET}_${RUN_ID}.jsonl"
    local log_file="${LOG_DIR}/generate_${model}_${DATASET}_${RUN_ID}.log"

    local cmd=(
        "${PYTHON_BIN}" "${AGORA_DIR}/scripts/generate_with_memory.py"
        --dataset "${DATASET}"
        --model "${model}"
        --temperature "${TEMPERATURE}"
        --output "${output_file}"
        --memory_top_k 0
        --parallel "${GEN_PARALLEL}"
        --max_problems "${MAX_PROBLEMS}"
        --no_auto_debug
    )

    "${cmd[@]}" | tee "${log_file}"
}

run_debate() {
    local log_file="${LOG_DIR}/debate_${DATASET}_${RUN_ID}.log"

    local cmd=(
        "${PYTHON_BIN}" "${AGORA_DIR}/scripts/run_memory_debate.py"
        --modelA "${MODEL_A}"
        --modelB "${MODEL_B}"
        --results_dir "${GENERATION_DIR}"
        --datasets "${DATASET}"
        --output_root "${DEBATE_DIR}"
        --max_rounds "${MAX_ROUNDS}"
        --temperature "${TEMPERATURE}"
        --debate_workers "${DEBATE_WORKERS}"
        --execute_workers "${EVAL_WORKERS}"
        --max_problems "${MAX_PROBLEMS}"
        --timeout "${TIMEOUT}"
        --tolerance "${TOLERANCE}"
        --relative_tolerance
        --disable_debate_memory
        --execute_disable_debug_memory
    )

    "${cmd[@]}" | tee "${log_file}"
}

echo "============================================================"
echo "Agora-Opt Quick Start"
echo "============================================================"
echo "Dataset:      ${DATASET}"
echo "Max problems: ${MAX_PROBLEMS}"
echo "Model A:      ${MODEL_A}"
echo "Model B:      ${MODEL_B}"
echo "Memory:       disabled"
echo "Results:      ${RESULTS_DIR}"
echo "============================================================"
echo

run_generation "${MODEL_A}"
echo
run_generation "${MODEL_B}"
echo
run_debate

echo
echo "Quick start completed."
echo "Generation outputs: ${GENERATION_DIR}"
echo "Debate outputs:     ${DEBATE_DIR}"
