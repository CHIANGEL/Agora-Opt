#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
AGORA_DIR="${PROJECT_ROOT}/code/Agora-Opt"
DATA_DIR="${PROJECT_ROOT}/data/benchmarks"
RESULTS_ROOT="${PROJECT_ROOT}/results/experiments/5.3.2_impact_of_debate_rounds"
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
    echo "Please set LLM_API_KEY and LLM_API_BASE_URL before running this experiment."
    exit 1
fi

MODEL_A="${MODEL_A:-gpt-4o}"
MODEL_B="${MODEL_B:-deepseek-chat}"
TEMPERATURE="${TEMPERATURE:-0.01}"
MAX_PROBLEMS="${MAX_PROBLEMS:-}"
GEN_PARALLEL="${GEN_PARALLEL:-32}"
EVAL_WORKERS="${EVAL_WORKERS:-64}"
DEBATE_WORKERS="${DEBATE_WORKERS:-16}"
TIMEOUT="${TIMEOUT:-90}"
TOLERANCE="${TOLERANCE:-0.05}"
MEMORY_TOP_K="${MEMORY_TOP_K:-3}"
DEBATE_MEMORY_TOP_K="${DEBATE_MEMORY_TOP_K:-2}"
DEBUG_CASE_TOP_K="${DEBUG_CASE_TOP_K:-3}"
MAX_RETRIES="${MAX_RETRIES:-5}"

MEMORY_DIR="${MEMORY_DIR:-${AGORA_DIR}/memory_storage}"
DEBUG_CASE_MEMORY_DIR="${DEBUG_CASE_MEMORY_DIR:-${AGORA_DIR}/debug_case_memory}"
DEBATE_MEMORY_DIR="${DEBATE_MEMORY_DIR:-${AGORA_DIR}/debate_memory_storage}"

GENERATION_DIR="${RESULTS_ROOT}/${RUN_ID}/generation"
DEBATE_ROOT="${RESULTS_ROOT}/${RUN_ID}/debate"
LOG_DIR="${RESULTS_ROOT}/${RUN_ID}/logs"
RUNTIME_DIR="${RESULTS_ROOT}/${RUN_ID}/runtime"
DEBUG_MEMORY_PATH="${DEBUG_MEMORY_PATH:-${RUNTIME_DIR}/debug_memory.jsonl}"

DATASETS=(
    "NL4OPT"
    "EasyLP"
    "ComplexLP"
    "IndustryOR"
    "OPT-Principled"
)

ROUNDS=(0 1 2 3)

export PYTHONPATH="${AGORA_DIR}/src:${PYTHONPATH:-}"

mkdir -p "${GENERATION_DIR}" "${DEBATE_ROOT}" "${LOG_DIR}" "${RUNTIME_DIR}"
touch "${DEBUG_MEMORY_PATH}"

require_path() {
    local path="$1"
    if [[ ! -e "${path}" ]]; then
        echo "Required path not found: ${path}"
        exit 1
    fi
}

require_path "${AGORA_DIR}/scripts/generate_with_memory.py"
require_path "${AGORA_DIR}/scripts/run_memory_debate.py"
require_path "${MEMORY_DIR}"
require_path "${DEBUG_CASE_MEMORY_DIR}"
require_path "${DEBATE_MEMORY_DIR}"
require_path "${DATA_DIR}"

for dataset in "${DATASETS[@]}"; do
    require_path "${DATA_DIR}/${dataset}.jsonl"
done

run_generation() {
    local model="$1"
    local dataset="$2"
    local output_file="${GENERATION_DIR}/${model}_${dataset}_${RUN_ID}.jsonl"
    local log_file="${LOG_DIR}/generate_${model}_${dataset}_${RUN_ID}.log"

    echo "------------------------------------------------------------"
    echo "Generation"
    echo "Model:   ${model}"
    echo "Dataset: ${dataset}"
    echo "Output:  ${output_file}"
    echo "Log:     ${log_file}"
    echo "------------------------------------------------------------"

    local cmd=(
        "${PYTHON_BIN}" "${AGORA_DIR}/scripts/generate_with_memory.py"
        --dataset "${dataset}"
        --model "${model}"
        --temperature "${TEMPERATURE}"
        --output "${output_file}"
        --memory_dir "${MEMORY_DIR}"
        --memory_top_k "${MEMORY_TOP_K}"
        --parallel "${GEN_PARALLEL}"
        --execution_timeout "${TIMEOUT}"
        --debug_memory_path "${DEBUG_MEMORY_PATH}"
        --debug_case_memory_dir "${DEBUG_CASE_MEMORY_DIR}"
        --debug_case_memory_top_k "${DEBUG_CASE_TOP_K}"
        --max_retries "${MAX_RETRIES}"
    )

    if [[ -n "${MAX_PROBLEMS}" ]]; then
        cmd+=(--max_problems "${MAX_PROBLEMS}")
    fi

    "${cmd[@]}" | tee "${log_file}"
    echo
}

run_debate_round() {
    local round="$1"
    local dataset="$2"
    local round_root="${DEBATE_ROOT}/round_${round}"
    local log_file="${LOG_DIR}/debate_round_${round}_${dataset}_${RUN_ID}.log"

    mkdir -p "${round_root}"

    echo "------------------------------------------------------------"
    echo "Debate"
    echo "Round:   ${round}"
    echo "Dataset: ${dataset}"
    echo "Output:  ${round_root}"
    echo "Log:     ${log_file}"
    echo "------------------------------------------------------------"

    local cmd=(
        "${PYTHON_BIN}" "${AGORA_DIR}/scripts/run_memory_debate.py"
        --modelA "${MODEL_A}"
        --modelB "${MODEL_B}"
        --results_dir "${GENERATION_DIR}"
        --datasets "${dataset}"
        --output_root "${round_root}"
        --max_rounds "${round}"
        --temperature "${TEMPERATURE}"
        --debate_workers "${DEBATE_WORKERS}"
        --execute_workers "${EVAL_WORKERS}"
        --timeout "${TIMEOUT}"
        --tolerance "${TOLERANCE}"
        --relative_tolerance
        --execute_memory_dir "${MEMORY_DIR}"
        --execute_debug_memory_path "${DEBUG_MEMORY_PATH}"
        --debate_memory_dir "${DEBATE_MEMORY_DIR}"
        --debate_memory_top_k "${DEBATE_MEMORY_TOP_K}"
    )

    if [[ -n "${MAX_PROBLEMS}" ]]; then
        cmd+=(--max_problems "${MAX_PROBLEMS}")
    fi

    "${cmd[@]}" | tee "${log_file}"
    echo
}

echo "============================================================"
echo "5.3.2 Impact of Debate Rounds"
echo "============================================================"
echo "Project root:      ${PROJECT_ROOT}"
echo "Results root:      ${RESULTS_ROOT}/${RUN_ID}"
echo "Model A:           ${MODEL_A}"
echo "Model B:           ${MODEL_B}"
echo "Datasets:          ${DATASETS[*]}"
echo "Rounds:            ${ROUNDS[*]}"
echo "Generation workers:${GEN_PARALLEL}"
echo "Debate workers:    ${DEBATE_WORKERS}"
echo "Eval workers:      ${EVAL_WORKERS}"
echo "============================================================"
echo

echo "Stage 1/2: generating initial solutions once"
echo
for model in "${MODEL_A}" "${MODEL_B}"; do
    for dataset in "${DATASETS[@]}"; do
        run_generation "${model}" "${dataset}"
    done
done

echo "Stage 2/2: running debate rounds"
echo
for round in "${ROUNDS[@]}"; do
    for dataset in "${DATASETS[@]}"; do
        run_debate_round "${round}" "${dataset}"
    done
done

echo "============================================================"
echo "Impact-of-rounds experiment completed"
echo "Generation outputs: ${GENERATION_DIR}"
echo "Debate outputs:     ${DEBATE_ROOT}"
echo "Logs:               ${LOG_DIR}"
echo "============================================================"
