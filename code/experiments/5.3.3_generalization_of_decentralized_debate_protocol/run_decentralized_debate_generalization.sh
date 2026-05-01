#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
AGORA_DIR="${PROJECT_ROOT}/code/Agora-Opt"
RESULTS_ROOT="${PROJECT_ROOT}/results/experiments/5.3.3_generalization_of_decentralized_debate_protocol"
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

TEMPERATURE="${TEMPERATURE:-0.01}"
MAX_ROUNDS="${MAX_ROUNDS:-3}"
MAX_PROBLEMS="${MAX_PROBLEMS:-}"
DEBATE_WORKERS="${DEBATE_WORKERS:-16}"
EVAL_WORKERS="${EVAL_WORKERS:-64}"
TIMEOUT="${TIMEOUT:-90}"
TOLERANCE="${TOLERANCE:-0.05}"
DEBATE_MEMORY_TOP_K="${DEBATE_MEMORY_TOP_K:-2}"

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

# These identifiers are used both as:
# 1. the staged filename prefixes consumed by run_memory_debate.py
# 2. the --modelA / --modelB values passed into the debate stage
#
# Replace them with the actual runnable model identifiers accepted by your backend
# when you are ready to execute the experiment end-to-end.
AGENTIC_MODEL_A="${AGENTIC_MODEL_A:-optimus-gpt4o}"
AGENTIC_MODEL_B="${AGENTIC_MODEL_B:-cafa-gpt4o}"
TRAINING_MODEL_A="${TRAINING_MODEL_A:-orlm-8b}"
TRAINING_MODEL_B="${TRAINING_MODEL_B:-steporlm-8b}"

# Fill in these paths when the corresponding initial-solution outputs and memory assets are ready.
#
# Agentic pair: OptiMUS + CAFA
: "${AGENTIC_SOLUTION_MEMORY_DIR:=}"
: "${AGENTIC_DEBUG_MEMORY_PATH:=}"
: "${AGENTIC_DEBATE_MEMORY_DIR:=}"

# Training-centric pair: ORLM + StepORLM
: "${TRAINING_SOLUTION_MEMORY_DIR:=}"
: "${TRAINING_DEBUG_MEMORY_PATH:=}"
: "${TRAINING_DEBATE_MEMORY_DIR:=}"

export PYTHONPATH="${AGORA_DIR}/src:${PYTHONPATH:-}"

require_path() {
    local path="$1"
    if [[ ! -e "${path}" ]]; then
        echo "Required path not found: ${path}"
        exit 1
    fi
}

to_env_key() {
    local raw="$1"
    echo "${raw}" | tr '[:lower:]-' '[:upper:]_' | tr -cd 'A-Z0-9_'
}

require_env_path() {
    local env_name="$1"
    local path="${!env_name:-}"
    if [[ -z "${path}" ]]; then
        echo "Missing required environment variable: ${env_name}"
        exit 1
    fi
    if [[ ! -f "${path}" ]]; then
        echo "File referenced by ${env_name} does not exist: ${path}"
        exit 1
    fi
}

require_env_dir() {
    local env_name="$1"
    local path="${!env_name:-}"
    if [[ -z "${path}" ]]; then
        echo "Missing required environment variable: ${env_name}"
        exit 1
    fi
    if [[ ! -d "${path}" ]]; then
        echo "Directory referenced by ${env_name} does not exist: ${path}"
        exit 1
    fi
}

stage_pair_inputs() {
    local pair_name="$1"
    local model_a="$2"
    local model_b="$3"
    local prefix_a="$4"
    local prefix_b="$5"
    local staging_dir="$6"
    local stage_ts="$7"

    mkdir -p "${staging_dir}"

    for dataset in "${DATASETS[@]}"; do
        local dataset_key
        dataset_key="$(to_env_key "${dataset}")"
        local env_a="${prefix_a}_INITIAL_SOLUTION_${dataset_key}"
        local env_b="${prefix_b}_INITIAL_SOLUTION_${dataset_key}"

        require_env_path "${env_a}"
        require_env_path "${env_b}"

        local source_a="${!env_a}"
        local source_b="${!env_b}"
        local target_a="${staging_dir}/${model_a}_${dataset}_${stage_ts}.jsonl"
        local target_b="${staging_dir}/${model_b}_${dataset}_${stage_ts}.jsonl"

        ln -sfn "${source_a}" "${target_a}"
        ln -sfn "${source_b}" "${target_b}"

        echo "[${pair_name}] staged ${dataset}"
        echo "  A: ${env_a} -> ${target_a}"
        echo "  B: ${env_b} -> ${target_b}"
    done
}

run_pair() {
    local pair_name="$1"
    local model_a="$2"
    local model_b="$3"
    local prefix_a="$4"
    local prefix_b="$5"
    local solution_memory_env="$6"
    local debug_memory_env="$7"
    local debate_memory_env="$8"
    local solution_memory_dir="${!solution_memory_env:-}"
    local debug_memory_path="${!debug_memory_env:-}"
    local debate_memory_dir="${!debate_memory_env:-}"
    local pair_root="${RESULTS_ROOT}/${RUN_ID}/${pair_name}"
    local staging_dir="${pair_root}/staged_initial_solutions"
    local output_root="${pair_root}/debate"
    local log_dir="${pair_root}/logs"
    local log_file="${log_dir}/run_memory_debate.log"
    local stage_ts="${RUN_ID}"

    mkdir -p "${output_root}" "${log_dir}"

    echo "============================================================"
    echo "5.3.3 Generalization of Decentralized Debate Protocol"
    echo "Pair:        ${pair_name}"
    echo "Model A:     ${model_a}"
    echo "Model B:     ${model_b}"
    echo "Results:     ${pair_root}"
    echo "Datasets:     ${DATASETS[*]}"
    echo "Solution memory: ${solution_memory_dir}"
    echo "Debug memory:    ${debug_memory_path}"
    echo "Debate memory:${debate_memory_dir}"
    echo "============================================================"

    require_env_dir "${solution_memory_env}"
    require_env_path "${debug_memory_env}"
    require_env_dir "${debate_memory_env}"

    stage_pair_inputs "${pair_name}" "${model_a}" "${model_b}" "${prefix_a}" "${prefix_b}" "${staging_dir}" "${stage_ts}"

    local cmd=(
        "${PYTHON_BIN}" "${AGORA_DIR}/scripts/run_memory_debate.py"
        --modelA "${model_a}"
        --modelB "${model_b}"
        --results_dir "${staging_dir}"
        --datasets "${DATASETS[@]}"
        --output_root "${output_root}"
        --max_rounds "${MAX_ROUNDS}"
        --temperature "${TEMPERATURE}"
        --debate_workers "${DEBATE_WORKERS}"
        --execute_workers "${EVAL_WORKERS}"
        --timeout "${TIMEOUT}"
        --tolerance "${TOLERANCE}"
        --relative_tolerance
        --execute_memory_dir "${solution_memory_dir}"
        --execute_debug_memory_path "${debug_memory_path}"
        --debate_memory_dir "${debate_memory_dir}"
        --debate_memory_top_k "${DEBATE_MEMORY_TOP_K}"
    )

    if [[ -n "${MAX_PROBLEMS}" ]]; then
        cmd+=(--max_problems "${MAX_PROBLEMS}")
    fi

    "${cmd[@]}" | tee "${log_file}"
    echo
}

require_path "${AGORA_DIR}/scripts/run_memory_debate.py"

run_pair \
    "agentic_optimus_plus_cafa" \
    "${AGENTIC_MODEL_A}" \
    "${AGENTIC_MODEL_B}" \
    "OPTIMUS" \
    "CAFA" \
    "AGENTIC_SOLUTION_MEMORY_DIR" \
    "AGENTIC_DEBUG_MEMORY_PATH" \
    "AGENTIC_DEBATE_MEMORY_DIR"

run_pair \
    "training_orlm_plus_steporlm" \
    "${TRAINING_MODEL_A}" \
    "${TRAINING_MODEL_B}" \
    "ORLM" \
    "STEPORLM" \
    "TRAINING_SOLUTION_MEMORY_DIR" \
    "TRAINING_DEBUG_MEMORY_PATH" \
    "TRAINING_DEBATE_MEMORY_DIR"

echo "============================================================"
echo "5.3.3 experiment template completed"
echo "Results root: ${RESULTS_ROOT}/${RUN_ID}"
echo "============================================================"
