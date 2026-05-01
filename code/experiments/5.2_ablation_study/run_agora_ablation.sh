#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
AGORA_DIR="${PROJECT_ROOT}/code/Agora-Opt"
DATA_DIR="${PROJECT_ROOT}/data/benchmarks"
ZERO_SHOT_DIR="${PROJECT_ROOT}/code/baseline/zero-shot-LLM"
RESULT_ROOT="${PROJECT_ROOT}/results/experiments/5.2_ablation_study"
RUN_ID_BASE="${RUN_ID_BASE:-$(date -u +"%Y%m%d_%H%M%S")}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

# API credential template.
: "${LLM_API_KEY:=PUT YOUR API KEY HERE}"
: "${LLM_API_BASE_URL:=PUT YOUR API URL HERE}"
export LLM_API_KEY
export LLM_API_BASE_URL

# Compatibility mirrors for different client code.
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
TIMEOUT="${TIMEOUT:-90}"
TOLERANCE="${TOLERANCE:-0.05}"
GEN_PARALLEL="${GEN_PARALLEL:-32}"
EVAL_WORKERS="${EVAL_WORKERS:-64}"
DEBATE_WORKERS="${DEBATE_WORKERS:-16}"
MAX_ROUNDS="${MAX_ROUNDS:-3}"
MAX_RETRIES="${MAX_RETRIES:-5}"
MEMORY_TOP_K_DEFAULT="${MEMORY_TOP_K_DEFAULT:-3}"
DEBUG_CASE_TOP_K_DEFAULT="${DEBUG_CASE_TOP_K_DEFAULT:-3}"
DEBATE_MEMORY_TOP_K_DEFAULT="${DEBATE_MEMORY_TOP_K_DEFAULT:-2}"
ZERO_SHOT_MAX_WORKERS="${ZERO_SHOT_MAX_WORKERS:-5}"
ZERO_SHOT_TIMEOUT_S="${ZERO_SHOT_TIMEOUT_S:-500}"
ZERO_SHOT_MAX_TOKENS="${ZERO_SHOT_MAX_TOKENS:-40000}"
ZERO_SHOT_MAXIMUM_RETRIES="${ZERO_SHOT_MAXIMUM_RETRIES:-5}"

MEMORY_DIR="${MEMORY_DIR:-${AGORA_DIR}/memory_storage}"
DEBUG_CASE_MEMORY_DIR="${DEBUG_CASE_MEMORY_DIR:-${AGORA_DIR}/debug_case_memory}"
DEBATE_MEMORY_DIR="${DEBATE_MEMORY_DIR:-${AGORA_DIR}/debate_memory_storage}"

DATASETS=(
    "ComplexLP"
    "IndustryOR"
    "OPT-Principled"
)

export PYTHONPATH="${AGORA_DIR}/src:${PYTHONPATH:-}"

require_path() {
    local path="$1"
    if [[ ! -e "${path}" ]]; then
        echo "Required path not found: ${path}"
        exit 1
    fi
}

require_path "${AGORA_DIR}/scripts/generate_with_memory.py"
require_path "${AGORA_DIR}/scripts/run_memory_debate.py"
require_path "${AGORA_DIR}/scripts/execute.py"
require_path "${ZERO_SHOT_DIR}/run_test.py"
require_path "${MEMORY_DIR}"
require_path "${DEBUG_CASE_MEMORY_DIR}"
require_path "${DEBATE_MEMORY_DIR}"

for dataset in "${DATASETS[@]}"; do
    require_path "${DATA_DIR}/${dataset}.jsonl"
done

run_generation_pass() {
    local variant_root="$1"
    local model="$2"
    local dataset="$3"
    local run_id="$4"
    local memory_top_k="$5"
    local debug_case_top_k="$6"
    local debug_memory_path="$7"
    local generation_dir="${variant_root}/generation"
    local log_dir="${variant_root}/logs"
    local output_file="${generation_dir}/${model}_${dataset}_${run_id}.jsonl"
    local log_file="${log_dir}/generate_${model}_${dataset}_${run_id}.log"

    mkdir -p "${generation_dir}" "${log_dir}"

    echo "------------------------------------------------------------"
    echo "Generation"
    echo "Variant root: ${variant_root}"
    echo "Model:        ${model}"
    echo "Dataset:      ${dataset}"
    echo "Memory top-k: ${memory_top_k}"
    echo "Debug top-k:  ${debug_case_top_k}"
    echo "Output:       ${output_file}"
    echo "------------------------------------------------------------"

    local cmd=(
        "${PYTHON_BIN}" "${AGORA_DIR}/scripts/generate_with_memory.py"
        --dataset "${dataset}"
        --model "${model}"
        --temperature "${TEMPERATURE}"
        --output "${output_file}"
        --memory_dir "${MEMORY_DIR}"
        --memory_top_k "${memory_top_k}"
        --parallel "${GEN_PARALLEL}"
        --execution_timeout "${TIMEOUT}"
        --debug_memory_path "${debug_memory_path}"
        --debug_case_memory_dir "${DEBUG_CASE_MEMORY_DIR}"
        --debug_case_memory_top_k "${debug_case_top_k}"
        --max_retries "${MAX_RETRIES}"
    )

    "${cmd[@]}" | tee "${log_file}"
    echo
}

run_debate_pass() {
    local variant_root="$1"
    local dataset="$2"
    local run_id="$3"
    local debug_memory_path="$4"
    local use_debate_memory="$5"
    local generation_dir="${variant_root}/generation"
    local debate_dir="${variant_root}/debate"
    local log_dir="${variant_root}/logs"
    local log_file="${log_dir}/debate_${dataset}_${run_id}.log"

    mkdir -p "${debate_dir}" "${log_dir}"

    echo "------------------------------------------------------------"
    echo "Debate"
    echo "Variant root: ${variant_root}"
    echo "Dataset:      ${dataset}"
    echo "Output:       ${debate_dir}"
    echo "------------------------------------------------------------"

    local cmd=(
        "${PYTHON_BIN}" "${AGORA_DIR}/scripts/run_memory_debate.py"
        --modelA "${MODEL_A}"
        --modelB "${MODEL_B}"
        --results_dir "${generation_dir}"
        --datasets "${dataset}"
        --output_root "${debate_dir}"
        --max_rounds "${MAX_ROUNDS}"
        --temperature "${TEMPERATURE}"
        --debate_workers "${DEBATE_WORKERS}"
        --execute_workers "${EVAL_WORKERS}"
        --timeout "${TIMEOUT}"
        --tolerance "${TOLERANCE}"
        --relative_tolerance
        --execute_memory_dir "${MEMORY_DIR}"
        --execute_debug_memory_path "${debug_memory_path}"
    )

    if [[ "${use_debate_memory}" == "true" ]]; then
        cmd+=(--debate_memory_dir "${DEBATE_MEMORY_DIR}" --debate_memory_top_k "${DEBATE_MEMORY_TOP_K_DEFAULT}")
    else
        cmd+=(--disable_debate_memory)
    fi

    "${cmd[@]}" | tee "${log_file}"
    echo
}

run_single_agent_eval() {
    local variant_root="$1"
    local dataset="$2"
    local run_id="$3"
    local debug_memory_path="$4"
    local generation_dir="${variant_root}/generation"
    local eval_root="${variant_root}/evaluation"
    local log_dir="${variant_root}/logs"
    local input_file="${generation_dir}/${MODEL_A}_${dataset}_${run_id}.jsonl"
    local eval_dir="${eval_root}/${dataset}_${run_id}"
    local log_file="${log_dir}/execute_${dataset}_${run_id}.log"

    mkdir -p "${eval_root}" "${log_dir}"

    echo "------------------------------------------------------------"
    echo "Single-agent evaluation"
    echo "Variant root: ${variant_root}"
    echo "Dataset:      ${dataset}"
    echo "Input:        ${input_file}"
    echo "Output:       ${eval_dir}"
    echo "------------------------------------------------------------"

    local cmd=(
        "${PYTHON_BIN}" "${AGORA_DIR}/scripts/execute.py"
        --input_file "${input_file}"
        --output_dir "${eval_dir}"
        --num_workers "${EVAL_WORKERS}"
        --timeout "${TIMEOUT}"
        --tolerance "${TOLERANCE}"
        --use_relative_tolerance
        --memory_dir "${MEMORY_DIR}"
        --debug_memory_path "${debug_memory_path}"
    )

    "${cmd[@]}" | tee "${log_file}"
    echo
}

run_agora_pair_variant() {
    local variant_name="$1"
    local memory_top_k="$2"
    local debug_case_top_k="$3"
    local use_debate_memory="$4"
    local run_id="${RUN_ID_BASE}_${variant_name}"
    local variant_root="${RESULT_ROOT}/${variant_name}"
    local runtime_dir="${variant_root}/runtime"
    local debug_memory_path="${runtime_dir}/debug_memory.jsonl"

    mkdir -p "${runtime_dir}"
    : > "${debug_memory_path}"

    echo "============================================================"
    echo "Running variant: ${variant_name}"
    echo "Run id:         ${run_id}"
    echo "Results root:   ${variant_root}"
    echo "============================================================"
    echo

    for model in "${MODEL_A}" "${MODEL_B}"; do
        for dataset in "${DATASETS[@]}"; do
            run_generation_pass "${variant_root}" "${model}" "${dataset}" "${run_id}" "${memory_top_k}" "${debug_case_top_k}" "${debug_memory_path}"
        done
    done

    for dataset in "${DATASETS[@]}"; do
        run_debate_pass "${variant_root}" "${dataset}" "${run_id}" "${debug_memory_path}" "${use_debate_memory}"
    done
}

run_single_agent_variant() {
    local variant_name="$1"
    local memory_top_k="$2"
    local debug_case_top_k="$3"
    local run_id="${RUN_ID_BASE}_${variant_name}"
    local variant_root="${RESULT_ROOT}/${variant_name}"
    local runtime_dir="${variant_root}/runtime"
    local debug_memory_path="${runtime_dir}/debug_memory.jsonl"

    mkdir -p "${runtime_dir}"
    : > "${debug_memory_path}"

    echo "============================================================"
    echo "Running variant: ${variant_name}"
    echo "Run id:         ${run_id}"
    echo "Results root:   ${variant_root}"
    echo "============================================================"
    echo

    for dataset in "${DATASETS[@]}"; do
        run_generation_pass "${variant_root}" "${MODEL_A}" "${dataset}" "${run_id}" "${memory_top_k}" "${debug_case_top_k}" "${debug_memory_path}"
        run_single_agent_eval "${variant_root}" "${dataset}" "${run_id}" "${debug_memory_path}"
    done
}

run_vanilla_gpt4o_variant() {
    local variant_name="$1"
    local variant_root="${RESULT_ROOT}/${variant_name}"
    local log_dir="${variant_root}/logs"

    mkdir -p "${variant_root}" "${log_dir}"

    echo "============================================================"
    echo "Running variant: ${variant_name}"
    echo "Results root:   ${variant_root}"
    echo "============================================================"
    echo

    for dataset in "${DATASETS[@]}"; do
        local dataset_path="${DATA_DIR}/${dataset}.jsonl"
        local log_file="${log_dir}/zero_shot_gpt4o_${dataset}.log"

        echo "------------------------------------------------------------"
        echo "Vanilla GPT-4o"
        echo "Dataset: ${dataset}"
        echo "Output:  ${variant_root}"
        echo "------------------------------------------------------------"

        "${PYTHON_BIN}" "${ZERO_SHOT_DIR}/run_test.py" \
            --model "gpt-4o" \
            --dataset "${dataset_path}" \
            --output-dir "${variant_root}" \
            --max-workers "${ZERO_SHOT_MAX_WORKERS}" \
            --timeout-s "${ZERO_SHOT_TIMEOUT_S}" \
            --maximum-retries "${ZERO_SHOT_MAXIMUM_RETRIES}" \
            --answer-rtol "${TOLERANCE}" \
            --max-tokens "${ZERO_SHOT_MAX_TOKENS}" | tee "${log_file}"
        echo
    done
}

echo "============================================================"
echo "Agora-Opt Ablation Study"
echo "============================================================"
echo "Project root: ${PROJECT_ROOT}"
echo "Results root: ${RESULT_ROOT}"
echo "Datasets:     ${DATASETS[*]}"
echo "============================================================"
echo

# Full Agora-Opt
run_agora_pair_variant "full_agora_opt" "${MEMORY_TOP_K_DEFAULT}" "${DEBUG_CASE_TOP_K_DEFAULT}" "true"

# Remove Debate Memory
run_agora_pair_variant "no_debate_memory" "${MEMORY_TOP_K_DEFAULT}" "${DEBUG_CASE_TOP_K_DEFAULT}" "false"

# Remove Debug Memory (implemented as disabling debug-case retrieval in generation)
run_agora_pair_variant "no_debug_memory" "${MEMORY_TOP_K_DEFAULT}" "0" "true"

# Remove Solution Memory
run_agora_pair_variant "no_solution_memory" "0" "${DEBUG_CASE_TOP_K_DEFAULT}" "true"

# Remove Debate (Single Agent) — use GPT-4o generation/evaluation without memory retrieval.
run_single_agent_variant "no_debate_single_agent" "0" "0"

# Remove Agent Team (Vanilla GPT-4o)
run_vanilla_gpt4o_variant "vanilla_gpt4o"

echo "============================================================"
echo "Ablation study completed"
echo "Results root: ${RESULT_ROOT}"
echo "============================================================"
