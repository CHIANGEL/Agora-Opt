#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
SRC_DIR="${PROJECT_ROOT}/src"
export PYTHONPATH="${SRC_DIR}:${PYTHONPATH:-}"
GENERATE_CLI="${PROJECT_ROOT}/scripts/generate_with_memory.py"

# Test self-healing mechanism with a small sample
# This will test the full pipeline with just 3 problems

echo "================================================"
echo "🧪 Testing Self-Healing Mechanism"
echo "================================================"
echo ""

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate or-debate

# Test parameters
MODEL="deepseek-chat"
DATASET="IndustryOR"
MAX_PROBLEMS=3
OUTPUT_DIR="${PROJECT_ROOT}/test_output"
OUTPUT_FILE="${OUTPUT_DIR}/test_self_healing_$(date +%Y%m%d_%H%M%S).jsonl"
MEMORY_DIR="${PROJECT_ROOT}/memory_storage"
MAX_RETRIES=3

mkdir -p "${OUTPUT_DIR}"

echo "Configuration:"
echo "  Model:        ${MODEL}"
echo "  Dataset:      ${DATASET}"
echo "  Max problems: ${MAX_PROBLEMS}"
echo "  Max retries:  ${MAX_RETRIES}"
echo "  Output:       ${OUTPUT_FILE}"
echo ""

# Run generation with self-healing
set +e
python "${GENERATE_CLI}" \
    --dataset "${DATASET}" \
    --model "${MODEL}" \
    --max_problems "${MAX_PROBLEMS}" \
    --output "${OUTPUT_FILE}" \
    --memory_dir "${MEMORY_DIR}" \
    --memory_top_k 3 \
    --parallel 1 \
    --max_retries "${MAX_RETRIES}" \
    --execution_timeout 60
EXIT_CODE=$?
set -e


if [ ${EXIT_CODE} -ne 0 ]; then
    echo ""
    echo "❌ Test failed with exit code ${EXIT_CODE}"
    exit 1
fi

echo ""
echo "================================================"
echo "📊 Test Results"
echo "================================================"

if [ -f "${OUTPUT_FILE}" ]; then
    TOTAL=$(wc -l < "${OUTPUT_FILE}")
    echo "Total problems processed: ${TOTAL}"
    
    # Count successes
    SUCCESS=$(grep -c '"execution_status": "success"' "${OUTPUT_FILE}" 2>/dev/null || echo 0)
    echo "Successful executions:    ${SUCCESS}"
    
    # Count with retries
    RETRIED=$(grep -c '"total_attempts": [2-9]' "${OUTPUT_FILE}" 2>/dev/null || echo 0)
    echo "Problems that used retry: ${RETRIED}"
    
    # Show sample result
    echo ""
    echo "Sample result (problem 1):"
    head -1 "${OUTPUT_FILE}" | python -m json.tool | grep -E '"id"|"execution_status"|"total_attempts"|"self_healing_enabled"'
    
    echo ""
    echo "✅ Test completed successfully!"
    echo "Full results saved to: ${OUTPUT_FILE}"
else
    echo "❌ Output file not found: ${OUTPUT_FILE}"
    exit 1
fi
