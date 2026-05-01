#!/bin/bash

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
OPEN_ROOT="$(cd "${PROJECT_ROOT}/../.." && pwd)"
SRC_DIR="${PROJECT_ROOT}/src"
export PYTHONPATH="${SRC_DIR}:${PYTHONPATH:-}"

# Generate and Evaluate - Combined pipeline for generation + evaluation
# Usage: ./run_generate_and_evaluate.sh [model_name] [max_problems] [num_workers] [timeout] [tolerance] [dataset_name]
#
# Environment Variables:
#   REFRESH_DEBUG_MEMORY - Set to "false" to disable auto-backup and clearing of debug memory (default: true)
#   RUN_ALL_BENCHMARKS - Set to "true" to run all benchmarks in ./data/benchmarks/ (default: true)
#   USE_HF_OFFLINE - Set to "false" to allow downloading models from Hugging Face (default: true)
#   PARALLEL_BENCHMARKS - Set to "true" to run benchmarks in parallel (default: true)
#   MAX_PARALLEL_JOBS - Maximum number of parallel jobs (default: 4)
#   DATASET_NAME - Dataset to run when RUN_ALL_BENCHMARKS=false (default: IndustryOR)
#   EMBEDDING_MODEL - Optional embedding model name or local path passed to memory retrieval
#
# Example:
#   ./run_generate_and_evaluate.sh                    # Run with default settings (all benchmarks, offline mode, parallel)
#   RUN_ALL_BENCHMARKS=false ./run_generate_and_evaluate.sh  # Run single dataset
#   RUN_ALL_BENCHMARKS=false ./run_generate_and_evaluate.sh gpt-4o 100 64 90 0.05 OPT-Principled
#   USE_HF_OFFLINE=false ./run_generate_and_evaluate.sh  # Allow downloading models
#   REFRESH_DEBUG_MEMORY=false ./run_generate_and_evaluate.sh  # Run without refreshing debug memory
#   PARALLEL_BENCHMARKS=false ./run_generate_and_evaluate.sh  # Run sequentially
#   MAX_PARALLEL_JOBS=2 ./run_generate_and_evaluate.sh  # Limit to 2 parallel jobs

MODEL=${1:-"gpt-4o"}
MAX_PROBLEMS=${2:-1000}
NUM_WORKERS=${3:-100}
TIMEOUT=${4:-60}
TOLERANCE=${5:-0.05}

# Configuration: Auto-backup and clear debug memory before running
# Set to "false" to disable this feature
REFRESH_DEBUG_MEMORY=${REFRESH_DEBUG_MEMORY:-true}

# Configuration: Run all benchmarks or single dataset
RUN_ALL_BENCHMARKS=${RUN_ALL_BENCHMARKS:-true}

# Configuration: Use offline mode for Hugging Face (avoid network calls)
# Set to "false" if you need to download models for the first time
USE_HF_OFFLINE=${USE_HF_OFFLINE:-true}

# Configuration: Run benchmarks in parallel
# Set to "true" to enable concurrent datasets (default: sequential datasets)
PARALLEL_BENCHMARKS=${PARALLEL_BENCHMARKS:-false}

# Configuration: Maximum number of parallel jobs
# Adjust based on your system resources
MAX_PARALLEL_JOBS=${MAX_PARALLEL_JOBS:-4}

# Default single dataset
DEFAULT_DATASET=${DATASET_NAME:-${6:-"IndustryOR"}}
# DEFAULT_DATASET="ComplexOR"
TEMPERATURE=${TEMPERATURE:-0.01}
MEMORY_DIR="${PROJECT_ROOT}/memory_storage"
MEMORY_TOP_K=${MEMORY_TOP_K:-3}
PARALLEL=${PARALLEL:-128}
MAIN_TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="${OPEN_ROOT}/results/Agora-Opt/generate_and_evaluate"
MAX_RETRIES=${MAX_RETRIES:-5}
BENCHMARKS_DIR="${PROJECT_ROOT}/../../data/benchmarks"
EMBEDDING_MODEL=${EMBEDDING_MODEL:-}

GENERATE_CLI="${PROJECT_ROOT}/scripts/generate_with_memory.py"
EXECUTE_CLI="${PROJECT_ROOT}/scripts/execute.py"

if [ -d "${BENCHMARKS_DIR}" ]; then
    BENCHMARKS_DIR="$(cd "${BENCHMARKS_DIR}" && pwd)"
elif [ -d "${PROJECT_ROOT}/clean_benchmarks" ]; then
    BENCHMARKS_DIR="$(cd "${PROJECT_ROOT}/clean_benchmarks" && pwd)"
elif [ -d "${PROJECT_ROOT}/../clean_benchmarks" ]; then
    BENCHMARKS_DIR="$(cd "${PROJECT_ROOT}/../clean_benchmarks" && pwd)"
fi

# Create output directory
mkdir -p "${OUTPUT_DIR}"

ensure_or_debate_env() {
    if [ "${CONDA_DEFAULT_ENV:-}" = "or-debate" ] && command -v python >/dev/null 2>&1; then
        return 0
    fi

    if ! command -v conda >/dev/null 2>&1; then
        echo "❌ conda command not found. Please install Conda or activate the or-debate environment manually."
        return 1
    fi

    local conda_bin
    local conda_base
    conda_bin="$(command -v conda)"
    conda_base="$(cd "$(dirname "${conda_bin}")/.." && pwd)"

    if [ -f "${conda_base}/etc/profile.d/conda.sh" ]; then
        # shellcheck disable=SC1090
        source "${conda_base}/etc/profile.d/conda.sh"
    else
        eval "$("${conda_bin}" shell.bash hook)"
    fi

    conda activate or-debate
}

# ============================================
# Function: Backup and Clear Debug Memory
# ============================================
backup_debug_memory() {
    if [ "${REFRESH_DEBUG_MEMORY}" = "true" ]; then
        DEBUG_MEMORY_FILE="${MEMORY_DIR}/debug_memory.jsonl"
        BACKUP_DIR="${MEMORY_DIR}/backups/${MAIN_TIMESTAMP}"

        if [ -f "${DEBUG_MEMORY_FILE}" ]; then
            echo "================================================"
            echo "🗂️  Backing up debug memory..."
            echo "================================================"
            
            # Create backup directory
            mkdir -p ${BACKUP_DIR}
            
            # Copy debug_memory.jsonl to backup
            cp "${DEBUG_MEMORY_FILE}" "${BACKUP_DIR}/debug_memory.jsonl"
            
            # Get file size and line count
            FILE_SIZE=$(du -h "${DEBUG_MEMORY_FILE}" | cut -f1)
            LINE_COUNT=$(wc -l < "${DEBUG_MEMORY_FILE}")
            
            echo "✅ Backed up debug memory:"
            echo "  Location: ${BACKUP_DIR}/debug_memory.jsonl"
            echo "  Size:     ${FILE_SIZE}"
            echo "  Lines:    ${LINE_COUNT}"
            
            # Clear the original file
            > "${DEBUG_MEMORY_FILE}"
            echo "✅ Cleared original debug memory file"
            echo ""
        else
            echo "ℹ️  No debug memory file found, skipping backup"
            echo ""
        fi
    else
        echo "ℹ️  Debug memory refresh is disabled (REFRESH_DEBUG_MEMORY=false)"
        echo ""
    fi
}

normalize_dataset_name() {
    local dataset_name="$1"
    dataset_name="${dataset_name%.jsonl}"
    case "${dataset_name}" in
        ComplexLP_clean) echo "ComplexLP" ;;
        EasyLP_clean) echo "EasyLP" ;;
        IndustryOR_clean|IndustryOR_v2|IndustryOR_fixedV2|IndustryOR_fixedV2_clean) echo "IndustryOR" ;;
        NL4Opt|NL4Opt_clean|NL4OPT_clean) echo "NL4OPT" ;;
        NLP4LP_clean) echo "NLP4LP" ;;
        ComplexOR_clean) echo "ComplexOR" ;;
        ReSocratic_clean) echo "ReSocratic" ;;
        combined|combined_dataset|OPT-Principled_clean) echo "OPT-Principled" ;;
        *) echo "${dataset_name}" ;;
    esac
}

DEFAULT_DATASET="$(normalize_dataset_name "${DEFAULT_DATASET}")"

# ============================================
# Function: Run single dataset (core logic)
# ============================================
process_dataset() {
    local DATASET_NAME
    DATASET_NAME="$(normalize_dataset_name "$1")"
    local TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
    local OUTPUT_FILE="${OUTPUT_DIR}/${MODEL}_${DATASET_NAME}_${TIMESTAMP}.jsonl"
    local EVAL_FILE="${OUTPUT_DIR}/${MODEL}_${DATASET_NAME}_eval_${TIMESTAMP}.jsonl"
    local EVAL_REPORT="${EVAL_FILE}/evaluation_report.json"
    
    echo ""
    echo "╔════════════════════════════════════════════════╗"
    echo "║  Processing Dataset: ${DATASET_NAME}"
    echo "╚════════════════════════════════════════════════╝"
    echo ""
    
    # ============================================
    # STEP 1: Generation
    # ============================================
    echo "================================================"
    echo "📝 STEP 1/2: Generating code with memory..."
    echo "================================================"
    echo "Dataset: ${DATASET_NAME}"
    echo ""
    
    local generate_args=(
        --dataset "${DATASET_NAME}"
        --model "${MODEL}"
        --temperature "${TEMPERATURE}"
        --max_problems "${MAX_PROBLEMS}"
        --memory_dir "${MEMORY_DIR}"
        --memory_top_k "${MEMORY_TOP_K}"
        --parallel "${PARALLEL}"
        --output "${OUTPUT_FILE}"
        --max_retries "${MAX_RETRIES}"
        --execution_timeout 60
    )

    if [ -n "${EMBEDDING_MODEL}" ]; then
        generate_args+=(--embedding_model "${EMBEDDING_MODEL}")
    fi

    python "${GENERATE_CLI}" "${generate_args[@]}"

    EXIT_CODE=$?
    
    if [ ${EXIT_CODE} -ne 0 ]; then
        echo ""
        echo "❌ Generation failed for ${DATASET_NAME} with exit code ${EXIT_CODE}"
        return 1
    fi
    
    echo ""
    echo "✅ Generation completed for ${DATASET_NAME}!"
    echo ""
    
    # Show generation summary
    if [ -f "${OUTPUT_FILE}" ]; then
        TOTAL=$(wc -l < ${OUTPUT_FILE})
        SUCCESS=$(grep -c '"status": "success"' "${OUTPUT_FILE}" 2>/dev/null || true)
        if [ -z "${SUCCESS}" ]; then
            SUCCESS=0
        fi
        echo "📊 Generation Summary:"
        echo "  Total problems: ${TOTAL}"
        echo "  Successful:     ${SUCCESS}"

        if [ "${SUCCESS}" -eq 0 ]; then
            echo ""
            echo "❌ Generation produced zero successful solutions for ${DATASET_NAME}"
            echo "   Refusing to continue with an incomplete run."
            return 1
        fi
    fi
    
    echo ""
    
    # ============================================
    # STEP 2: Evaluation
    # ============================================
    echo "================================================"
    echo "🔍 STEP 2/2: Executing and evaluating..."
    echo "================================================"
    echo ""
    
    local execute_args=(
        --input_file "${OUTPUT_FILE}"
        --output_dir "${EVAL_FILE}"
        --num_workers "${NUM_WORKERS}"
        --timeout "${TIMEOUT}"
        --tolerance "${TOLERANCE}"
        --use_relative_tolerance
    )

    if [ -n "${EMBEDDING_MODEL}" ]; then
        execute_args+=(--embedding_model "${EMBEDDING_MODEL}")
    fi

    python "${EXECUTE_CLI}" "${execute_args[@]}"
    EXIT_CODE=$?
    
    if [ ${EXIT_CODE} -ne 0 ]; then
        echo ""
        echo "❌ Evaluation failed for ${DATASET_NAME} with exit code ${EXIT_CODE}"
        return 1
    fi
    
    echo ""
    echo "✅ Evaluation completed for ${DATASET_NAME}!"
    echo ""
    
    # Show evaluation report if exists
    if [ -f "${EVAL_REPORT}" ]; then
        echo "📊 Evaluation Results for ${DATASET_NAME}:"
        cat "${EVAL_REPORT}" | jq '{
            accuracy: .accuracy,
            correct: .correct,
            total: .total_problems,
            status_counts: .status_counts
        }' 2>/dev/null || cat "${EVAL_REPORT}"
        echo ""
        
        # Store results for final summary (with lock for parallel execution)
        ACCURACY=$(cat "${EVAL_REPORT}" | jq -r '.accuracy' 2>/dev/null || echo "N/A")
        CORRECT=$(cat "${EVAL_REPORT}" | jq -r '.correct' 2>/dev/null || echo "N/A")
        TOTAL_PROBS=$(cat "${EVAL_REPORT}" | jq -r '.total_problems' 2>/dev/null || echo "N/A")
        
        # Use lock to safely append to results file (fallback to simple append if flock not available)
        RESULTS_LOCK="${OUTPUT_DIR}/batch_results_${MAIN_TIMESTAMP}.lock"
        if command -v flock >/dev/null 2>&1; then
            (
                flock -x 200
                echo "${DATASET_NAME}|${ACCURACY}|${CORRECT}|${TOTAL_PROBS}|${EVAL_FILE}" >> "${OUTPUT_DIR}/batch_results_${MAIN_TIMESTAMP}.txt"
            ) 200>"${RESULTS_LOCK}"
        else
            # Fallback: use simple append (may have race condition but unlikely with small writes)
            echo "${DATASET_NAME}|${ACCURACY}|${CORRECT}|${TOTAL_PROBS}|${EVAL_FILE}" >> "${OUTPUT_DIR}/batch_results_${MAIN_TIMESTAMP}.txt"
        fi
    fi
    
    echo "================================================"
    echo ""
    
    if [ -f "${EVAL_REPORT}" ]; then
        return 0
    else
        return 1
    fi
}

# ============================================
# Function: Run single dataset (internal, supports logging)
# ============================================
run_single_dataset_internal() {
    local DATASET_NAME=$1
    local LOG_FILE=$2
    local STREAM_OUTPUT=${3:-false}
    
    if [ "${STREAM_OUTPUT}" = "true" ]; then
        process_dataset "${DATASET_NAME}" |& tee "${LOG_FILE}"
        local EXIT_CODE=${PIPESTATUS[0]}
        return ${EXIT_CODE}
    else
        process_dataset "${DATASET_NAME}" > "${LOG_FILE}" 2>&1
        return $?
    fi
}

# ============================================
# Function: Run single dataset (wrapper for sequential execution)
# ============================================
run_single_dataset() {
    local DATASET_NAME=$1
    local STREAM_OUTPUT=${2:-false}
    local LOG_FILE="${OUTPUT_DIR}/${DATASET_NAME}_${MAIN_TIMESTAMP}.log"
    
    run_single_dataset_internal "${DATASET_NAME}" "${LOG_FILE}" "${STREAM_OUTPUT}"
    local EXIT_CODE=$?
    
    # Display output only when we did not already stream it live
    if [ "${STREAM_OUTPUT}" != "true" ]; then
        cat "${LOG_FILE}"
    fi
    
    return ${EXIT_CODE}
}

# ============================================
# Main Execution
# ============================================

echo "================================================"
echo "🚀 Generate + Evaluate Pipeline"
echo "================================================"
echo "Model:        ${MODEL}"
echo "Max problems: ${MAX_PROBLEMS}"
echo "Temperature:  ${TEMPERATURE}"
echo "Memory dir:   ${MEMORY_DIR}"
echo "Memory Top-K: ${MEMORY_TOP_K}"
if [ -n "${EMBEDDING_MODEL}" ]; then
    echo "Embedding:    ${EMBEDDING_MODEL}"
else
    echo "Embedding:    MemoryBank default"
fi
echo "Parallel:     ${PARALLEL}"
echo "Refresh Memory: ${REFRESH_DEBUG_MEMORY}"
echo "Run All Benchmarks: ${RUN_ALL_BENCHMARKS}"
echo "HF Offline:   ${USE_HF_OFFLINE}"
echo "Parallel Benchmarks: ${PARALLEL_BENCHMARKS}"
if [ "${PARALLEL_BENCHMARKS}" = "true" ]; then
    echo "Max Parallel Jobs: ${MAX_PARALLEL_JOBS}"
fi
echo ""
echo "Eval Workers: ${NUM_WORKERS}"
echo "Eval Timeout: ${TIMEOUT}s"
echo "Tolerance:    ${TOLERANCE} (relative)"
echo ""
echo "Max retries:  ${MAX_RETRIES}"
echo "================================================"
echo ""

# Activate environment
ensure_or_debate_env || exit 1

# Set Hugging Face offline mode if enabled
if [ "${USE_HF_OFFLINE}" = "true" ]; then
    echo "ℹ️  Hugging Face offline mode enabled (using local cache)"
    export HF_HUB_OFFLINE=1
    export TRANSFORMERS_OFFLINE=1
    export HF_DATASETS_OFFLINE=1
else
    echo "ℹ️  Hugging Face online mode (may download models if needed)"
fi
echo ""

# Backup and clear debug memory (only once at the beginning)
backup_debug_memory

# ============================================
# Run benchmarks
# ============================================
if [ "${RUN_ALL_BENCHMARKS}" = "true" ]; then
    if [ "${PARALLEL_BENCHMARKS}" = "true" ]; then
        echo "================================================"
        echo "🔄 Running ALL benchmarks in PARALLEL"
        echo "================================================"
    else
        echo "================================================"
        echo "🔄 Running ALL benchmarks SEQUENTIALLY"
        echo "================================================"
    fi
    echo ""
    
    # Define benchmark dataset names in specified order (without .jsonl extension)
    # Modify this array to change the execution order
    BENCHMARK_NAMES=(
        "NL4OPT"
        "EasyLP"
        "ComplexLP"
        "NLP4LP"
        "ComplexOR"
        "IndustryOR"
        "ReSocratic"
        "OPT-Principled"
    )
    
    # Count total benchmarks
    TOTAL_BENCHMARKS=${#BENCHMARK_NAMES[@]}
    FAILED=0
    SKIPPED=0
    
    echo "Total benchmarks to process: ${TOTAL_BENCHMARKS}"
    echo ""
    echo "Execution order:"
    for i in "${!BENCHMARK_NAMES[@]}"; do
        echo "  $((i+1)). ${BENCHMARK_NAMES[$i]}"
    done
    echo ""
    
    # Initialize batch results file
    echo "Dataset|Accuracy|Correct|Total|Output" > "${OUTPUT_DIR}/batch_results_${MAIN_TIMESTAMP}.txt"
    
    # Create lock file for parallel execution
    RESULTS_LOCK="${OUTPUT_DIR}/batch_results_${MAIN_TIMESTAMP}.lock"
    touch "${RESULTS_LOCK}"
    
    # Process benchmarks (parallel or sequential)
    if [ "${PARALLEL_BENCHMARKS}" = "true" ]; then
        # Parallel execution
        declare -a PIDS=()
        declare -a DATASET_NAMES=()
        CURRENT_JOBS=0
        
        for DATASET_NAME in "${BENCHMARK_NAMES[@]}"; do
            BENCHMARK_FILE="${BENCHMARKS_DIR}/${DATASET_NAME}.jsonl"
            
            # Check if file exists
            if [ ! -f "${BENCHMARK_FILE}" ]; then
                echo "⚠️  File not found: ${BENCHMARK_FILE}"
                echo "   Skipping ${DATASET_NAME}..."
                SKIPPED=$((SKIPPED + 1))
                continue
            fi
            
            # Wait for available slot if max jobs reached
            while true; do
                # Count running jobs
                CURRENT_JOBS=0
                for PID in "${PIDS[@]}"; do
                    if kill -0 ${PID} 2>/dev/null; then
                        CURRENT_JOBS=$((CURRENT_JOBS + 1))
                    fi
                done
                
                # Break if we have available slots
                if [ ${CURRENT_JOBS} -lt ${MAX_PARALLEL_JOBS} ]; then
                    break
                fi
                
                # Wait a bit before checking again
                sleep 1
            done
            
            # Start job in background
            LOG_FILE="${OUTPUT_DIR}/${DATASET_NAME}_${MAIN_TIMESTAMP}.log"
            echo "🚀 Starting ${DATASET_NAME} (log: ${LOG_FILE})"
            
            (
                run_single_dataset_internal "${DATASET_NAME}" "${LOG_FILE}"
                EXIT_CODE=$?
                if [ ${EXIT_CODE} -ne 0 ]; then
                    echo "[${DATASET_NAME}] ❌ Failed with exit code ${EXIT_CODE}" >> "${OUTPUT_DIR}/failures_${MAIN_TIMESTAMP}.txt"
                else
                    echo "[${DATASET_NAME}] ✅ Completed successfully" >> "${OUTPUT_DIR}/success_${MAIN_TIMESTAMP}.txt"
                fi
            ) &
            
            PID=$!
            PIDS+=(${PID})
            DATASET_NAMES+=("${DATASET_NAME}")
        done
        
        # Wait for all jobs to complete
        echo ""
        echo "⏳ Waiting for all jobs to complete..."
        echo ""
        
        for i in "${!PIDS[@]}"; do
            PID=${PIDS[$i]}
            DATASET_NAME=${DATASET_NAMES[$i]}
            wait ${PID}
            EXIT_CODE=$?
            if [ ${EXIT_CODE} -ne 0 ]; then
                FAILED=$((FAILED + 1))
                echo "⚠️  ${DATASET_NAME} failed with exit code ${EXIT_CODE}"
            fi
        done
        
        # Clean up lock file
        rm -f "${RESULTS_LOCK}"
        
        echo ""
        echo "================================================"
        echo "📋 Individual Job Logs:"
        echo "================================================"
        for DATASET_NAME in "${BENCHMARK_NAMES[@]}"; do
            LOG_FILE="${OUTPUT_DIR}/${DATASET_NAME}_${MAIN_TIMESTAMP}.log"
            if [ -f "${LOG_FILE}" ]; then
                echo ""
                echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
                echo "  ${DATASET_NAME} - Log File: ${LOG_FILE}"
                echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
                tail -20 "${LOG_FILE}"
            fi
        done
        echo ""
        
    else
        # Sequential execution
        CURRENT=0
        for DATASET_NAME in "${BENCHMARK_NAMES[@]}"; do
            CURRENT=$((CURRENT + 1))
            BENCHMARK_FILE="${BENCHMARKS_DIR}/${DATASET_NAME}.jsonl"
            
            echo ""
            echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
            echo "  Progress: ${CURRENT}/${TOTAL_BENCHMARKS}"
            echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
            
            # Check if file exists
            if [ ! -f "${BENCHMARK_FILE}" ]; then
                echo "⚠️  File not found: ${BENCHMARK_FILE}"
                echo "   Skipping..."
                SKIPPED=$((SKIPPED + 1))
                continue
            fi
            
            run_single_dataset "${DATASET_NAME}" true
            
            if [ $? -ne 0 ]; then
                FAILED=$((FAILED + 1))
                echo "⚠️  Failed to process ${DATASET_NAME}, continuing..."
            fi
            
            echo ""
        done
        
        # Clean up lock file
        rm -f "${RESULTS_LOCK}"
    fi
    
    # ============================================
    # Final Summary for All Benchmarks
    # ============================================
    echo ""
    echo "================================================"
    echo "🎉 All Benchmarks Complete!"
    echo "================================================"
    echo ""
    echo "Summary:"
    echo "  Total benchmarks: ${TOTAL_BENCHMARKS}"
    echo "  Successful:       $((TOTAL_BENCHMARKS - FAILED - SKIPPED))"
    echo "  Failed:           ${FAILED}"
    echo "  Skipped:          ${SKIPPED}"
    echo ""
    echo "📊 Detailed Results:"
    echo "================================================"
    
    # Display formatted results table
    if [ -f "${OUTPUT_DIR}/batch_results_${MAIN_TIMESTAMP}.txt" ]; then
        echo ""
        printf "%-35s | %-10s | %-10s | %-10s\n" "Dataset" "Accuracy" "Correct" "Total"
        echo "--------------------------------------------------------------------------------"
        tail -n +2 "${OUTPUT_DIR}/batch_results_${MAIN_TIMESTAMP}.txt" | while IFS='|' read -r dataset accuracy correct total output; do
            printf "%-35s | %-10s | %-10s | %-10s\n" "${dataset}" "${accuracy}" "${correct}" "${total}"
        done
        echo ""
        echo "📁 Full results saved to: ${OUTPUT_DIR}/batch_results_${MAIN_TIMESTAMP}.txt"
    fi
    
    echo ""
    echo "================================================"
    
else
    # Run single dataset mode
    echo "================================================"
    echo "📝 Running single dataset: ${DEFAULT_DATASET}"
    echo "================================================"
    echo ""

    BENCHMARK_FILE="${BENCHMARKS_DIR}/${DEFAULT_DATASET}.jsonl"
    if [ ! -f "${BENCHMARK_FILE}" ]; then
        echo "❌ Dataset file not found: ${BENCHMARK_FILE}"
        exit 1
    fi
    
    run_single_dataset "${DEFAULT_DATASET}" true
    
    if [ $? -ne 0 ]; then
        echo ""
        echo "❌ Pipeline failed"
        exit 1
    fi
    
    echo ""
    echo "🎉 Pipeline Complete!"
fi

echo ""
echo "✨ All done! Check the results above."
echo ""
