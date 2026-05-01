#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
AGORA_DIR="${PROJECT_ROOT}/code/Agora-Opt"
DATA_DIR="${PROJECT_ROOT}/data/benchmarks"
METHOD_NAME="${METHOD_NAME:-Agora-Opt}"
RESULTS_DIR="${PROJECT_ROOT}/results/${METHOD_NAME}"
GENERATION_DIR="${RESULTS_DIR}/generation"
DEBATE_DIR="${RESULTS_DIR}/debate"
LOG_DIR="${RESULTS_DIR}/logs"
RUNTIME_DIR="${RESULTS_DIR}/runtime"
PYTHON_BIN="${PYTHON_BIN:-python3}"
RUN_ID="${RUN_ID:-$(date -u +"%Y%m%d_%H%M%S")}"

# API credential template.
: "${LLM_API_KEY:=PUT YOUR API KEY HERE}"
: "${LLM_API_BASE_URL:=PUT YOUR API URL HERE}"
export LLM_API_KEY
export LLM_API_BASE_URL

# Compatibility mirrors for code that may still look for other env var names.
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
TEMPERATURE="${TEMPERATURE:-0.01}"
MAX_PROBLEMS="${MAX_PROBLEMS:-}"
GEN_PARALLEL="${GEN_PARALLEL:-32}"
EVAL_WORKERS="${EVAL_WORKERS:-64}"
DEBATE_WORKERS="${DEBATE_WORKERS:-16}"
MAX_ROUNDS="${MAX_ROUNDS:-3}"
TIMEOUT="${TIMEOUT:-90}"
TOLERANCE="${TOLERANCE:-0.05}"
MEMORY_TOP_K="${MEMORY_TOP_K:-3}"
DEBATE_MEMORY_TOP_K="${DEBATE_MEMORY_TOP_K:-2}"
DEBUG_CASE_TOP_K="${DEBUG_CASE_TOP_K:-3}"
MAX_RETRIES="${MAX_RETRIES:-5}"

MEMORY_DIR="${MEMORY_DIR:-${AGORA_DIR}/memory_storage}"
DEBUG_CASE_MEMORY_DIR="${DEBUG_CASE_MEMORY_DIR:-${AGORA_DIR}/debug_case_memory}"
DEBATE_MEMORY_DIR="${DEBATE_MEMORY_DIR:-${AGORA_DIR}/debate_memory_storage}"
DEBUG_MEMORY_PATH="${DEBUG_MEMORY_PATH:-${RUNTIME_DIR}/debug_memory.jsonl}"

DATASETS=(
    "NL4OPT"
    "EasyLP"
    "ComplexLP"
    "NLP4LP"
    "ComplexOR"
    "IndustryOR"
    "ReSocratic"
    "OPT-Principled"
)

export PYTHONPATH="${AGORA_DIR}/src:${PYTHONPATH:-}"

mkdir -p "${GENERATION_DIR}" "${DEBATE_DIR}" "${LOG_DIR}" "${RUNTIME_DIR}"
touch "${DEBUG_MEMORY_PATH}"

# Maintain legacy-looking compatibility entrypoints for the main Agora run only.
# This keeps the original stage-1 / stage-2 directory semantics visible without
# forcing new results to live inside the source tree.
if [[ "${METHOD_NAME}" == "Agora-Opt" ]]; then
    ln -sfn "${GENERATION_DIR}" "${AGORA_DIR}/generated_with_memory"
    ln -sfn "${DEBATE_DIR}" "${AGORA_DIR}/debate_runs"
fi

require_path() {
    local path="$1"
    if [[ ! -e "${path}" ]]; then
        echo "Required path not found: ${path}"
        exit 1
    fi
}

require_path "${AGORA_DIR}/scripts/generate_with_memory.py"
require_path "${AGORA_DIR}/scripts/run_memory_debate.py"
require_path "${DATA_DIR}"
require_path "${MEMORY_DIR}"
require_path "${DEBUG_CASE_MEMORY_DIR}"
require_path "${DEBATE_MEMORY_DIR}"

for dataset in "${DATASETS[@]}"; do
    require_path "${DATA_DIR}/${dataset}.jsonl"
done

echo "============================================================"
echo "Agora-Opt Main-Table Reproduction"
echo "============================================================"
echo "Project root:        ${PROJECT_ROOT}"
echo "Agora source:        ${AGORA_DIR}"
echo "Benchmarks:          ${DATA_DIR}"
echo "Results root:        ${RESULTS_DIR}"
echo "Run id:              ${RUN_ID}"
echo "Model A:             ${MODEL_A}"
echo "Model B:             ${MODEL_B}"
echo "Datasets:            ${DATASETS[*]}"
echo "Generation workers:  ${GEN_PARALLEL}"
echo "Debate workers:      ${DEBATE_WORKERS}"
echo "Eval workers:        ${EVAL_WORKERS}"
echo "Execution timeout:   ${TIMEOUT}"
echo "Tolerance:           ${TOLERANCE}"
echo "============================================================"
echo

run_generation() {
    local model="$1"
    local dataset="$2"
    local output_file="${GENERATION_DIR}/${model}_${dataset}_${RUN_ID}.jsonl"
    local log_file="${LOG_DIR}/generate_${model}_${dataset}_${RUN_ID}.log"

    echo "------------------------------------------------------------"
    echo "Stage 1: generating single-model solutions"
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

run_debate() {
    local dataset="$1"
    local log_file="${LOG_DIR}/debate_${dataset}_${RUN_ID}.log"

    echo "------------------------------------------------------------"
    echo "Stage 2: debate + consensus evaluation"
    echo "Dataset: ${dataset}"
    echo "Output:  ${DEBATE_DIR}"
    echo "Log:     ${log_file}"
    echo "------------------------------------------------------------"

    local cmd=(
        "${PYTHON_BIN}" "${AGORA_DIR}/scripts/run_memory_debate.py"
        --modelA "${MODEL_A}"
        --modelB "${MODEL_B}"
        --results_dir "${GENERATION_DIR}"
        --datasets "${dataset}"
        --output_root "${DEBATE_DIR}"
        --max_rounds "${MAX_ROUNDS}"
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

echo "Starting Stage 1/2: single-model generation"
echo
for model in "${MODEL_A}" "${MODEL_B}"; do
    for dataset in "${DATASETS[@]}"; do
        run_generation "${model}" "${dataset}"
    done
done

echo "Starting Stage 2/2: debate and evaluation"
echo
for dataset in "${DATASETS[@]}"; do
    run_debate "${dataset}"
done

echo "============================================================"
echo "Agora-Opt run completed"
echo "Generation outputs: ${GENERATION_DIR}"
echo "Debate outputs:     ${DEBATE_DIR}"
echo "Logs:               ${LOG_DIR}"
echo "============================================================"
