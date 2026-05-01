#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
RUN_AGORA_SH="${PROJECT_ROOT}/code/scripts/run_agora.sh"
EXPERIMENT_NAME="5.1_compatibility_backbone_llms"
RESULT_PREFIX="experiments/${EXPERIMENT_NAME}"
RUN_ID_BASE="${RUN_ID_BASE:-$(date -u +"%Y%m%d_%H%M%S")}"

# API credential template.
: "${LLM_API_KEY:=PUT YOUR API KEY HERE}"
: "${LLM_API_BASE_URL:=PUT YOUR API URL HERE}"
export LLM_API_KEY
export LLM_API_BASE_URL

# Compatibility mirrors for callers that use other env names.
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

if [[ ! -f "${RUN_AGORA_SH}" ]]; then
    echo "Missing dependency script: ${RUN_AGORA_SH}"
    exit 1
fi

run_variant() {
    local variant_label="$1"
    local model_a="$2"
    local model_b="$3"
    local variant_key="$4"
    local run_id="${RUN_ID_BASE}_${variant_key}"

    echo "============================================================"
    echo "Running experiment variant: ${variant_label}"
    echo "Model A: ${model_a}"
    echo "Model B: ${model_b}"
    echo "Run id:  ${run_id}"
    echo "Results: ./results/${RESULT_PREFIX}/${variant_key}"
    echo "============================================================"
    echo

    METHOD_NAME="${RESULT_PREFIX}/${variant_key}" \
    MODEL_A="${model_a}" \
    MODEL_B="${model_b}" \
    RUN_ID="${run_id}" \
    bash "${RUN_AGORA_SH}"

    echo
}

run_variant "Agora-Opt (GPT-4o + DeepSeek-V3)" "gpt-4o" "deepseek-chat" "gpt4o_plus_deepseekv3"
run_variant "Agora-Opt (DeepSeek-V3 + Gemini-2.5-Pro)" "deepseek-chat" "gemini-2.5-pro" "deepseekv3_plus_gemini25pro"
run_variant "Agora-Opt (GPT-4o + Gemini-2.5-Pro)" "gpt-4o" "gemini-2.5-pro" "gpt4o_plus_gemini25pro"

echo "============================================================"
echo "Experiment completed: ${EXPERIMENT_NAME}"
echo "Results root: ./results/${RESULT_PREFIX}"
echo "============================================================"
