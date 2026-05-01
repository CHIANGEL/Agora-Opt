#!/bin/bash

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
echo "Project directory: $PROJECT_DIR"

: "${LLM_API_KEY:=PUT YOUR API KEY HERE}"
: "${LLM_API_BASE_URL:=PUT YOUR API URL HERE}"
export LLM_API_KEY
export LLM_API_BASE_URL

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

# --- Models and Datasets ---
MODELS=(
    "OpenAI-o3"
    "OpenAI-o2"
    "Gemini-1.5-Pro"
    "Gemini-1.5-Flash"
    "Claude-3-Opus"
    "Claude-3-Sonnet"
    "Qwen2-72B-Instruct"
    "Mistral-Large"
    "Mistral-Medium"
    "Mistral-Small"
    "Mixtral-8x22B-Instruct-v0.1"
    "Mixtral-8x7B-Instruct-v0.1"
    "Llama-3-70B-Instruct"
    "Llama-3-8B-Instruct"
    "gemma-2-27b-it"
    "gemma-2-9b-it"
)

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

# --- Execution ---
CODE_DIR="$PROJECT_DIR/code"
BASELINE_DIR="$CODE_DIR/baseline/zero-shot-LLM"
BENCHMARK_DIR="$PROJECT_DIR/data/benchmarks"
RESULTS_DIR="$PROJECT_DIR/results/zero-shot-LLM"

# Ensure results directory exists
mkdir -p "$RESULTS_DIR"

# Run the baseline for each model and dataset
for model in "${MODELS[@]}"; do
    for dataset_name in "${DATASETS[@]}"; do
        dataset_path="$BENCHMARK_DIR/$dataset_name.jsonl"
        
        if [ ! -f "$dataset_path" ]; then
            echo "Dataset not found: $dataset_path"
            continue
        fi

        echo "--------------------------------------------------"
        echo "Running model: $model on dataset: $dataset_name"
        echo "--------------------------------------------------"

        python3 "$BASELINE_DIR/run_test.py" \
            --model "$model" \
            --dataset "$dataset_path" \
            --output-dir "$RESULTS_DIR"

        # Check the exit code of the Python script
        echo
    done
done

echo "All zero-shot-LLM baseline runs complete."
