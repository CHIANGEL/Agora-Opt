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
    "gpt-4o"
    "gemini-2.5-pro"
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
BASELINE_DIR="$CODE_DIR/baseline/chain-of-thought"
DATA_DIR="$PROJECT_DIR/data/benchmarks"
RESULTS_DIR="$PROJECT_DIR/results/chain-of-thought"

# Ensure results directory exists
mkdir -p "$RESULTS_DIR"

# Run the baseline for each model and dataset
for model in "${MODELS[@]}"; do
    for dataset_name in "${DATASETS[@]}"; do
        dataset_path="$DATA_DIR/$dataset_name.jsonl"
        
        if [ ! -f "$dataset_path" ]; then
            echo "Dataset not found: $dataset_path"
            continue
        fi

        echo "--------------------------------------------------"
        echo "Running model: $model on dataset: $dataset_name"
        echo "--------------------------------------------------"

        python3 "$BASELINE_DIR/run_cot.py" \
            --model "$model" \
            --dataset "$dataset_path" \
            --output-dir "$RESULTS_DIR"

        echo
    done
done

echo "All Chain-of-Thought baseline runs complete."
