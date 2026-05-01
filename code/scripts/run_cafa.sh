#!/bin/bash

set -euo pipefail

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

# Resolve project paths
BASE_DIR=$(dirname "$(dirname "$0")")
CAFA_DIR="$BASE_DIR/baseline/CAFA"
DATA_DIR="$BASE_DIR/../data/benchmarks"
export PYTHONPATH="$CAFA_DIR:$PYTHONPATH"

# Define models and datasets
MODELS=("gpt-4o" "gemini-2.5-pro")
DATASETS=("NL4OPT" "EasyLP" "ComplexLP" "NLP4LP" "ComplexOR" "IndustryOR" "ReSocratic" "OPT-Principled")

# Run all model-dataset combinations
for model in "${MODELS[@]}"; do
    for dataset in "${DATASETS[@]}"; do
        echo "Running CAFA with model: $model on dataset: $dataset"
        python3 "$CAFA_DIR/CAFA_test.py" \
            --dataset "$DATA_DIR/$dataset.jsonl" \
            --model "$model" \
            --output_dir "$BASE_DIR/../results/CAFA"
    done
done
