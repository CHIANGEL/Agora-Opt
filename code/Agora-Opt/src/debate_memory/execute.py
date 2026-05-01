"""
Execute and evaluate generated Gurobi code
"""

import argparse
import json
import os
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

from .debug_utils import sanitize_code, save_debug_metadata, write_debug_suggestions

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
DEFAULT_MEMORY_DIR = PROJECT_ROOT / "memory_storage"
DEFAULT_GUIDELINES = DEFAULT_MEMORY_DIR / "category_guidelines.jsonl"
DEFAULT_DEBUG_MEMORY = DEFAULT_MEMORY_DIR / "debug_memory.jsonl"


def extract_objective_value(output: str) -> float:
    """
    Extract objective value from Gurobi output
    
    Args:
        output: stdout from Gurobi code execution
    
    Returns:
        Objective value as float, or None if not found
    """
    if not output or output.strip() == "":
        return None
    
    # Common patterns in Gurobi output
    patterns = [
        r'Optimal\s+[Oo]bjective[:\s]+([+-]?\d+\.?\d*(?:[eE][+-]?\d+)?)',
        r'Obj:\s*([+-]?\d+\.?\d*(?:[eE][+-]?\d+)?)',
        r'Best\s+objective\s+([+-]?\d+\.?\d*(?:[eE][+-]?\d+)?)',
        r'Objective\s+value:\s*([+-]?\d+\.?\d*(?:[eE][+-]?\d+)?)',
        r'OBJECTIVE_VALUE:\s*([+-]?\d+\.?\d*(?:[eE][+-]?\d+)?)',  # Our auto-added pattern
    ]

    for pattern in patterns:
        match = re.search(pattern, output, re.IGNORECASE)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                continue

    # Fallback: check common custom labels printed by prompts
    fallback_patterns = [
        r'Total\s+Cost[:\s]+([+-]?\d+\.?\d*(?:[eE][+-]?\d+)?)',
        r'Total\s+Profit[:\s]+([+-]?\d+\.?\d*(?:[eE][+-]?\d+)?)',
        r'Total\s+Net\s+Profit[:\s]+([+-]?\d+\.?\d*(?:[eE][+-]?\d+)?)',
        r'Total\s+Revenue[:\s]+([+-]?\d+\.?\d*(?:[eE][+-]?\d+)?)',
    ]

    for pattern in fallback_patterns:
        match = re.search(pattern, output, re.IGNORECASE)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                continue

    return None


def enhance_code_with_objective_print(code: str) -> str:
    """
    Add objective value printing to code if not already present
    
    This helps ensure we can extract the objective value even if
    the generated code doesn't print it explicitly.
    
    Note: Always adds a fallback print to handle cases where existing
    prints are conditional (e.g., inside if status == OPTIMAL blocks)
    """
    # Add code to print objective value (always add as a safety measure)
    enhancement_code = """
# Auto-added: Print objective value for evaluation (fallback)
try:
    # Try common variable names for Gurobi model
    if 'model' in dir():
        mdl = model
    elif 'm' in dir():
        mdl = m
    elif 'Model' in dir():
        mdl = Model
    else:
        mdl = None
    
    # Fallback: scan globals for a likely Gurobi model instance.
    # This helps when the generated code uses a non-standard variable name.
    if mdl is None:
        try:
            for _name, _val in list(globals().items()):
                try:
                    if hasattr(_val, 'objVal') and hasattr(_val, 'optimize'):
                        mdl = _val
                        break
                except Exception:
                    continue
        except Exception:
            pass

    if mdl is not None and hasattr(mdl, 'objVal'):
        try:
            obj_value = mdl.objVal
            print(f"OBJECTIVE_VALUE: {obj_value}")
        except:
            # Model might not have been solved yet
            pass
except:
    pass
"""
    
    return code + "\n" + enhancement_code


def execute_code(code: str, problem_id: int, output_dir: str, timeout: int = 60) -> Dict:
    """
    Execute Gurobi code and capture results
    
    Args:
        code: Python code to execute
        problem_id: Problem ID
        output_dir: Directory to save code files
        timeout: Execution timeout in seconds
    
    Returns:
        Dictionary with execution results
    """
    # Create output directory
    code_dir = os.path.join(output_dir, 'code')
    os.makedirs(code_dir, exist_ok=True)

    sanitized_code, debug_meta = sanitize_code(code, problem_id)
    code_enhanced = enhance_code_with_objective_print(sanitized_code)
    
    # Save code to file
    code_file = os.path.join(code_dir, f'problem_{problem_id}.py')
    with open(code_file, 'w', encoding='utf-8') as f:
        f.write(code_enhanced)

    # Persist debug metadata if anything noteworthy was detected
    save_debug_metadata(debug_meta, output_dir)
    
    # Execute code
    try:
        result = subprocess.run(
            [sys.executable, f'problem_{problem_id}.py'],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=code_dir
        )
        
        stdout = result.stdout
        stderr = result.stderr
        returncode = result.returncode
        
        if returncode == 0:
            obj_value = extract_objective_value(stdout)
            if obj_value is not None:
                return {
                    'status': 'success',
                    'objective_value': obj_value,
                    'stdout': stdout,
                    'stderr': stderr
                }
            else:
                return {
                    'status': 'success_no_objective',
                    'objective_value': None,
                    'stdout': stdout,
                    'stderr': stderr
                }
        else:
            return {
                'status': 'execution_error',
                'objective_value': None,
                'stdout': stdout,
                'stderr': stderr,
                'returncode': returncode
            }
            
    except subprocess.TimeoutExpired:
        return {
            'status': 'timeout',
            'objective_value': None,
            'stdout': '',
            'stderr': f'Execution timeout after {timeout} seconds'
        }
    except Exception as e:
        return {
            'status': 'error',
            'objective_value': None,
            'stdout': '',
            'stderr': str(e)
        }


def check_correctness(pred_obj: float, gt_obj: float, tolerance: float = 0.05, 
                     use_relative: bool = True) -> bool:
    """
    Check if predicted objective matches ground truth
    
    Args:
        pred_obj: Predicted objective value
        gt_obj: Ground truth objective value
        tolerance: Tolerance for comparison
        use_relative: Use relative tolerance if True, absolute if False
    
    Returns:
        True if values match within tolerance
    """
    if pred_obj is None or gt_obj is None:
        return False
    
    try:
        pred_obj = float(pred_obj)
        gt_obj = float(gt_obj)
        
        if gt_obj == 0:
            return abs(pred_obj) <= tolerance
        
        if use_relative:
            return abs((pred_obj - gt_obj) / gt_obj) <= tolerance
        else:
            return abs(pred_obj - gt_obj) <= tolerance
    except (ValueError, TypeError):
        return False


def evaluate_results(results: List[Dict], args) -> Dict:
    """
    Evaluate execution results
    
    Args:
        results: List of result dictionaries
        args: Command line arguments
    
    Returns:
        Evaluation report dictionary
    """
    total = len(results)
    correct = 0
    
    status_counts = defaultdict(int)
    correct_ids = []
    incorrect_details = []
    
    for result in results:
        status = result['execution_status']
        status_counts[status] += 1
        
        if status == 'success' and result['is_correct']:
            correct += 1
            correct_ids.append(result['id'])
        elif status == 'success' and not result['is_correct']:
            incorrect_details.append({
                'id': result['id'],
                'predicted': result['predicted_objective'],
                'ground_truth': result['ground_truth']
            })
    
    accuracy = correct / total if total > 0 else 0.0
    
    report = {
        'total_problems': total,
        'correct': correct,
        'accuracy': accuracy,
        'status_counts': dict(status_counts),
        'correct_ids': correct_ids,
        'incorrect_details': incorrect_details[:10],  # Save first 10 for reference
        'settings': {
            'tolerance': args.tolerance,
            'use_relative_tolerance': args.use_relative_tolerance,
            'timeout': args.timeout
        }
    }
    
    return report


def process_single_problem(gen_result, args):
    """Process a single problem (for parallel execution)"""
    problem_id = gen_result['id']
    code = gen_result['generated_code']
    gt_answer = gen_result.get('answer')
    
    if not code:
        result = {
            'id': problem_id,
            'execution_status': 'no_code',
            'predicted_objective': None,
            'ground_truth': gt_answer,
            'is_correct': False
        }
    else:
        exec_result = execute_code(code, problem_id, args.output_dir, args.timeout)
        
        pred_obj = exec_result['objective_value']
        is_correct = False
        
        if pred_obj is not None and gt_answer is not None:
            try:
                gt_obj = float(gt_answer)
                is_correct = check_correctness(
                    pred_obj, gt_obj, 
                    args.tolerance, 
                    args.use_relative_tolerance
                )
            except (ValueError, TypeError):
                is_correct = False
        
        result = {
            'id': problem_id,
            'execution_status': exec_result['status'],
            'predicted_objective': pred_obj,
            'ground_truth': gt_answer,
            'is_correct': is_correct,
            'stdout': exec_result['stdout'][:500] if args.save_output else '',
            'stderr': exec_result['stderr'][:500] if args.save_output else ''
        }
    
    return result


def main(args):
    # Load generated results
    if not os.path.exists(args.input_file):
        raise FileNotFoundError(f"Input file not found: {args.input_file}")
    
    with open(args.input_file, 'r', encoding='utf-8') as f:
        generated_results = [json.loads(line) for line in f if line.strip()]
    
    print(f"Loaded {len(generated_results)} generated results")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    id_to_problem = {record['id']: record for record in generated_results}

    debug_store = None
    memory_helper = None
    memory_bank = None
    if not args.disable_debug_memory:
        try:
            from .debug_memory import DebugMemoryStore
            from .memory_bank import MemoryBank
            from .memory_intelligence import MemoryIntelligence
        except ModuleNotFoundError as exc:
            print(
                f"⚠️  Debug-memory dependencies missing ({exc}). "
                "Continuing with --disable_debug_memory behavior."
            )
            args.disable_debug_memory = True
        else:
            debug_store = DebugMemoryStore(args.debug_memory_path)
            if args.category_guidelines_path:
                try:
                    memory_helper = MemoryIntelligence(args.category_guidelines_path)
                except Exception as exc:  # noqa: BLE001
                    print(f"Warning: failed to load category guidelines ({exc})")
            if args.memory_dir:
                try:
                    if args.embedding_model:
                        memory_bank = MemoryBank(args.memory_dir, embedding_model=args.embedding_model)
                    else:
                        memory_bank = MemoryBank(args.memory_dir)
                except Exception as exc:  # noqa: BLE001
                    print(f"Warning: failed to load memory bank from {args.memory_dir} ({exc})")
    
    # Execute and evaluate each result
    evaluation_results = []
    
    if args.num_workers > 1:
        # Parallel execution
        print(f"Using {args.num_workers} workers for parallel execution")
        with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
            # Submit all tasks
            future_to_problem = {
                executor.submit(process_single_problem, gen_result, args): gen_result
                for gen_result in generated_results
            }
            
            # Collect results with progress bar
            with tqdm(total=len(generated_results), desc="Executing") as pbar:
                for future in as_completed(future_to_problem):
                    try:
                        result = future.result()
                        evaluation_results.append(result)
                        status_symbol = '✓' if result['is_correct'] else '✗'
                        pbar.set_postfix_str(f"Problem {result['id']}: {status_symbol}")
                        pbar.update(1)
                    except Exception as e:
                        gen_result = future_to_problem[future]
                        print(f"\nError processing problem {gen_result['id']}: {e}")
                        evaluation_results.append({
                            'id': gen_result['id'],
                            'execution_status': 'error',
                            'predicted_objective': None,
                            'ground_truth': gen_result.get('answer'),
                            'is_correct': False,
                            'stdout': '',
                            'stderr': str(e)
                        })
                        pbar.update(1)
        
        # Sort results by ID to maintain order
        evaluation_results.sort(key=lambda x: x['id'])
    else:
        # Sequential execution (original behavior)
        for gen_result in generated_results:
            problem_id = gen_result['id']
            print(f"Processing problem {problem_id}...", end=' ')
            
            result = process_single_problem(gen_result, args)
            evaluation_results.append(result)
            
            status_symbol = '✓' if result['is_correct'] else '✗'
            print(f"{status_symbol} [{result['execution_status']}]")
    
    # Provide memory-aided suggestions for failures
    if not args.disable_debug_memory:
        for result in evaluation_results:
            status = result['execution_status']
            if status in ('execution_error', 'success_no_objective', 'timeout', 'no_code'):
                gen_result = id_to_problem.get(result['id'], {})
                description = gen_result.get('description', '')
                error_message = result.get('stderr') or result.get('stdout') or ''
                if not error_message:
                    if status == 'timeout':
                        error_message = 'Execution timeout'
                    elif status == 'no_code':
                        error_message = 'No code was generated for execution.'
                    elif status == 'success_no_objective':
                        error_message = 'Execution succeeded but no objective value was captured.'
                write_debug_suggestions(
                    problem_id=result['id'],
                    description=description,
                    error_message=error_message,
                    memory_helper=memory_helper,
                    memory_bank=memory_bank,
                    output_dir=args.output_dir,
                    status=status,
                    debug_store=debug_store,
                )

    # Generate evaluation report
    report = evaluate_results(evaluation_results, args)
    
    # Save detailed results
    results_file = os.path.join(args.output_dir, 'evaluation_results.jsonl')
    with open(results_file, 'w', encoding='utf-8') as f:
        for result in evaluation_results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
    
    # Save evaluation report
    report_file = os.path.join(args.output_dir, 'evaluation_report.json')
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    # Print summary
    print(f"\n{'='*60}")
    print("EVALUATION SUMMARY")
    print(f"{'='*60}")
    print(f"Total problems:  {report['total_problems']}")
    print(f"Correct:         {report['correct']}")
    print(f"Accuracy:        {report['accuracy']:.2%}")
    print(f"\nStatus breakdown:")
    for status, count in sorted(report['status_counts'].items()):
        print(f"  {status:20s}: {count:3d} ({count/report['total_problems']:.1%})")
    print(f"{'='*60}")
    print(f"\nResults saved to:")
    print(f"  {results_file}")
    print(f"  {report_file}")


def parse_args():
    parser = argparse.ArgumentParser(description="Execute and evaluate generated Gurobi code")
    
    parser.add_argument("--input_file", type=str, required=True,
                        help="Path to generated results JSONL file")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save execution results")
    parser.add_argument("--timeout", type=int, default=60,
                        help="Timeout for code execution (seconds)")
    parser.add_argument("--tolerance", type=float, default=0.05,
                        help="Tolerance for answer comparison")
    parser.add_argument("--use_relative_tolerance", action="store_true",
                        help="Use relative tolerance (default: absolute)")
    parser.add_argument("--save_output", action="store_true",
                        help="Save stdout/stderr in results")
    parser.add_argument("--num_workers", type=int, default=100,
                        help="Number of parallel workers for execution")
    parser.add_argument("--memory_dir", type=str, default=str(DEFAULT_MEMORY_DIR),
                        help="Path to episodic memory directory (used for debug suggestions)")
    parser.add_argument("--embedding_model", type=str, default=None,
                        help="Optional embedding model name or local path for debug-memory retrieval")
    parser.add_argument("--category_guidelines_path", type=str,
                        default=str(DEFAULT_GUIDELINES),
                        help="Path to category guideline JSONL file")
    parser.add_argument("--debug_memory_path", type=str,
                        default=str(DEFAULT_DEBUG_MEMORY),
                        help="Path to persistent debug memory JSONL file")
    parser.add_argument("--disable_debug_memory", action="store_true",
                        help="Disable memory-based debug suggestions")
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
