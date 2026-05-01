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
OPTIMUS_DIR="$BASE_DIR/baseline/OptiMUS"
DATA_DIR="$BASE_DIR/../data/benchmarks"
export PYTHONPATH="$OPTIMUS_DIR:$PYTHONPATH"

# Define models and datasets
MODELS=("gpt-4o" "gemini-2.5-pro")
DATASETS=("NL4OPT" "EasyLP" "ComplexLP" "NLP4LP" "ComplexOR" "IndustryOR" "ReSocratic" "OPT-Principled")

# Run all model-dataset combinations
for model in "${MODELS[@]}"; do
    for dataset in "${DATASETS[@]}"; do
        echo "Running OptiMUS with model: $model on dataset: $dataset"
        python3 "$OPTIMUS_DIR/experiment_accuracy.py" \
            --dataset "$DATA_DIR/$dataset.jsonl" \
            --num 1000 \
            --start 0 \
            --mode single \
            --base_model "$model" \
            --start_stage 0 \
            --temperature 0.01 \
            --output_dir "$BASE_DIR/../results/OptiMUS"
    done
done
