"""
Generate with Memory: Single solution generation enhanced by memory retrieval
Based on simple_rag/generate.py + memory enhancement
"""

import argparse
import json
import os
import re
from pathlib import Path
from collections import Counter
from typing import Dict, List, Optional
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# Import local utilities
from .llm import get_response
from .config import find_benchmark_path, get_prompt_template, normalize_dataset_name

# Import memory bank
from .memory_bank import MemoryBank
from .debug_memory import DebugMemoryStore
from .debug_executor import execute_generated_code, ExecutionResult

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MEMORY_DIR = PROJECT_ROOT / "memory_storage"
DEFAULT_DEBUG_MEMORY = DEFAULT_MEMORY_DIR / "debug_memory.jsonl"
DEFAULT_DEBUG_CASE_MEMORY = PROJECT_ROOT / "debug_case_memory"


class NoOpMemoryBank:
    """Memory-bank stub used when retrieval is explicitly disabled."""

    case_count = 0

    def retrieve_similar_cases(self, query: str, top_k: int = 0):
        return []

    def format_retrieved_cases_for_prompt(self, similar_cases):
        return ""


def load_dataset(dataset_name: str) -> List[Dict]:
    """
    Load dataset from the migrated benchmark directory layout.
    
    Args:
        dataset_name: Name of the dataset (e.g., "ComplexLP", "IndustryOR")
    
    Returns:
        List of problem dictionaries with 'description' and 'answer' fields
    """
    dataset_name = normalize_dataset_name(dataset_name)
    dataset_path = find_benchmark_path(PROJECT_ROOT, dataset_name)
    
    problems = []
    with dataset_path.open('r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            if line.strip():
                data = json.loads(line)
                # Map en_question to description if it exists
                if 'en_question' in data and 'description' not in data:
                    data['description'] = data['en_question']
                # Map en_answer to answer if it exists  
                if 'en_answer' in data and 'answer' not in data:
                    data['answer'] = data['en_answer']
                # Set id if not already present
                if 'id' not in data:
                    data['id'] = idx
                problems.append(data)
    
    print(f"Loaded {len(problems)} problems from {dataset_name}")
    return problems


def extract_python_code(text: str) -> str:
    """
    Extract Python code from LLM output
    Looks for code within <python>...</python> tags or ```python...``` blocks
    
    Args:
        text: LLM output text
    
    Returns:
        Extracted Python code
    """
    # Try to extract from <python>...</python> tags first
    pattern_xml = r'<python>(.*?)</python>'
    match = re.search(pattern_xml, text, re.DOTALL | re.IGNORECASE)
    if match:
        code = match.group(1).strip()
        # Remove markdown code fences if present
        code = re.sub(r'^```python\s*\n', '', code)
        code = re.sub(r'\n```\s*$', '', code)
        return code
    
    # Try to extract from ```python...``` blocks
    pattern_markdown = r'```python(.*?)```'
    match = re.search(pattern_markdown, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    
    # If no code blocks found, return empty string
    return ""


def _truncate_text(text: str, limit: int = 1200) -> str:
    if isinstance(text, bytes):
        text = text.decode("utf-8", errors="replace")
    snippet = (text or "").strip()
    if not snippet:
        return ""
    if len(snippet) <= limit:
        return snippet
    return snippet[:limit] + "\n... (truncated)"


def write_debug_report(
    problem_id: int,
    description: str,
    exec_result: ExecutionResult,
    base_output_dir: str,
) -> str:
    debug_dir = os.path.join(base_output_dir, "debug")
    os.makedirs(debug_dir, exist_ok=True)
    path = os.path.join(debug_dir, f"problem_{problem_id}_debug.md")

    stdout_snippet = _truncate_text(exec_result.stdout)
    stderr_snippet = _truncate_text(exec_result.stderr)

    lines = [
        f"# Debug Report for Problem {problem_id}",
        "",
        f"- **Status:** {exec_result.status}",
    ]
    if exec_result.code_path:
        rel_path = os.path.relpath(exec_result.code_path, base_output_dir)
        lines.append(f"- **Code path:** {rel_path}")
    if description:
        lines.extend(["", "## Description", description.strip()])
    if stdout_snippet:
        lines.extend(["", "## Stdout", "```", stdout_snippet, "```"])
    if stderr_snippet:
        lines.extend(["", "## Stderr", "```", stderr_snippet, "```"])
    if not stdout_snippet and not stderr_snippet:
        lines.extend(["", "## Logs", "_No logs captured._"])

    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")

    return path


def filter_perfect_matches(similar_cases: List[Dict], current_description: str, max_filter: int = 1) -> List[Dict]:
    """
    Filter out cases with identical description (test set leakage)
    At most max_filter cases will be removed (default: 1)
    
    Args:
        similar_cases: List of retrieved cases with scores
        current_description: The description of current problem to compare against
        max_filter: Maximum number of perfect matches to filter out (default: 1)
    
    Returns:
        Filtered list of cases
    """
    filtered = []
    filtered_count = 0
    
    for case in similar_cases:
        case_desc = case['case'].get('description', '')
        problem_id = case['case'].get('problem_id', '?')
        score = case.get('score', 0.0)
        
        # Compare descriptions directly (exact match)
        # At most filter max_filter identical cases
        if case_desc.strip() == current_description.strip() and filtered_count < max_filter:
            filtered_count += 1
            print(f"  ⚠️  Filtered: Case ID={problem_id}, similarity={score:.4f} (identical description, test set leakage)")
        else:
            filtered.append(case)
    
    if filtered_count > 0:
        print(f"  📊 Filtered {filtered_count} perfect match(es) (max: {max_filter}), {len(filtered)} cases remaining")
    
    return filtered


def refine_retrieved_cases_with_llm(
    similar_cases: List[Dict],
    current_problem_desc: str,
    model: str,
    temperature: float = 0.3
) -> str:
    """
    Use LLM to analyze ALL retrieved cases together and extract key insights
    
    This is a two-stage process:
    1. Retrieve similar cases (vector similarity)
    2. Use LLM to view ALL cases holistically and extract transferable insights
    
    Args:
        similar_cases: List of retrieved cases
        current_problem_desc: Current problem description
        model: Model name for analysis
        temperature: Temperature for analysis (slightly higher for creativity)
    
    Returns:
        Refined insights as a string
    """
    if not similar_cases:
        return ""
    
    # Build full cases content (no truncation - show everything to LLM)
    full_cases = ""
    for i, item in enumerate(similar_cases, 1):
        case = item['case']
        score = item['score']
        full_cases += f"\n{'='*70}\n"
        full_cases += f"Case {i} (Similarity Score: {score:.3f})\n"
        full_cases += f"{'='*70}\n\n"
        full_cases += f"**Problem Description:**\n{case['description']}\n\n"
        full_cases += f"**Complete Solution Code:**\n```python\n{case['solution_code']}\n```\n\n"
        full_cases += f"**Objective Value:** {case['objective_value']}\n"
        full_cases += f"**Status:** Correct ✓\n"
        full_cases += "\n"
    
    analysis_prompt = f"""You are an expert in optimization modeling. You will analyze multiple similar solved problems to extract **transferable insights** for a new problem.

## Current Problem to Solve:
{current_problem_desc}

## Retrieved Similar Cases (Complete):
{full_cases}

## Your Task:

Analyze ALL the cases above **holistically** and provide a structured analysis that will guide solving the current problem.

**Focus on:**

1. **Problem Type & Structure**: What category do these problems fall into? (e.g., production planning, resource allocation, scheduling, network flow)

2. **Common Modeling Patterns**: 
   - What decision variables are typically used?
   - What types of constraints appear repeatedly?
   - How are objectives typically formulated?

3. **Key Techniques & Tricks**:
   - Any specific Gurobi features? (e.g., `addConstrs`, `quicksum`, binary variables, `setParam`)
   - Modeling tricks? (e.g., big-M, indicator constraints, piecewise linear)
   - Data structure patterns? (e.g., dictionaries for indices, list comprehensions)

4. **Adaptation Guidance**:
   - What aspects of the current problem are similar to the retrieved cases?
   - What's different and requires new thinking?
   - Which parts of the solution approaches can be directly adapted?

**Output Format**: 
Provide a concise, actionable analysis (300-500 words) structured by the 4 points above. Be specific with code patterns and techniques, not just high-level descriptions.

**Important**: Extract **transferable knowledge**, not just summarize. Think about what the solver needs to know to adapt these solutions to the current problem."""

    try:
        analysis = get_response(analysis_prompt, model=model, temperature=temperature)
        return analysis
    except Exception as e:
        print(f"  ⚠️  Warning: Failed to refine cases with LLM: {e}")
        # Fallback: return empty string, will use original formatting
        return ""


def format_debug_cases_for_prompt(cases: List[Dict]) -> str:
    if not cases:
        return ""
    lines = ["# Retrieved Debug Guidance", ""]
    for idx, item in enumerate(cases, 1):
        case = item["case"]
        score = item.get("score")
        signature = case.get("metadata", {}).get("signature", "unknown")
        status = case.get("metadata", {}).get("status", "")
        lines.append(f"## Case {idx} (similarity {score:.3f})")
        lines.append(f"Signature: {signature} | Status: {status}")
        description = case.get("description", "").strip()
        if description:
            lines.append(description if len(description) < 800 else description[:800] + "\n...")
        lines.append("---")
    return "\n".join(lines).strip()


def build_error_feedback_prompt(
    exec_result: ExecutionResult,
    attempt_number: int,
    previous_code: str,
    debug_guidance: str = ""
) -> str:
    """
    Build a prompt with error feedback for code correction
    
    Args:
        exec_result: Execution result with error information
        attempt_number: Current attempt number
        previous_code: The code that failed
    
    Returns:
        Feedback prompt string
    """
    error_info = exec_result.stderr if exec_result.stderr else exec_result.stdout
    if not error_info:
        error_info = f"Status: {exec_result.status}"
    
    feedback = f"""
# Code Execution Failed - Attempt {attempt_number}

Your previous code failed to execute successfully. Here is the error information:

## Error Details:
```
{error_info}
```

## Your Previous Code:
```python
{previous_code}
```

## Instructions:
1. Carefully analyze the error message above
2. Identify the root cause of the error
3. Fix the code to resolve the issue
4. Common issues to check:
   - Variable indexing (e.g., accessing index 0 when valid indices start from 1)
   - Missing variable definitions
   - Incorrect constraint formulations
   - Type mismatches

Please provide the CORRECTED code in a ```python``` code block. Make sure to:
- Fix the specific error mentioned above
- Keep the overall structure and logic intact
- Ensure all variables are properly defined before use
"""
    if debug_guidance:
        feedback += f"\n\n# Historical Debug Guidance\n{debug_guidance}\n"
    return feedback


def generate_with_memory(
    problem_id: int,
    problem_desc: str,
    memory_bank: MemoryBank,
    model: str,
    temperature: float,
    top_k: int = 4,
    filter_perfect: bool = True,
    use_llm_refinement: bool = True,
    *,
    auto_debug: bool = True,
    execution_timeout: int = 120,
    debug_output_dir: Optional[str] = None,
    debug_store: Optional[DebugMemoryStore] = None,
    max_retries: int = 3,
    debug_case_bank: Optional[MemoryBank] = None,
    debug_case_top_k: int = 3
) -> Dict:
    """
    Generate solution with memory enhancement
    
    Args:
        problem_id: Problem ID
        problem_desc: Problem description
        memory_bank: Memory bank instance
        model: Model name
        temperature: Generation temperature
        top_k: Number of cases to retrieve (default: 4, will filter identical descriptions)
        filter_perfect: Whether to filter out identical description matches
        use_llm_refinement: Whether to use LLM to refine/summarize retrieved cases
        auto_debug: Execute generated code and capture debug information
        execution_timeout: Timeout (seconds) for executing generated code
        debug_output_dir: Directory for storing debug artifacts (code, suggestions)
        debug_store: Persistent store for debug experiences
    
    Returns:
        Dict with generation results
    """
    # Retrieve similar cases from memory
    similar_cases = memory_bank.retrieve_similar_cases(problem_desc, top_k=top_k)
    original_retrieved = len(similar_cases)
    
    # Filter out identical descriptions (test set leakage)
    if filter_perfect and similar_cases:
        similar_cases = filter_perfect_matches(similar_cases, problem_desc)
    
    # Prepare memory context
    memory_context = ""
    refined_insights = ""
    
    if similar_cases:
        if use_llm_refinement:
            # Use LLM to analyze and refine the retrieved cases
            print(f"  🧠 Using LLM to refine {len(similar_cases)} retrieved cases...")
            refined_insights = refine_retrieved_cases_with_llm(
                similar_cases, problem_desc, model, temperature=0.3
            )
            
            if refined_insights:
                memory_context = f"""# Insights from Similar Problems in Memory

Based on analysis of {len(similar_cases)} similar problems, here are key insights:

{refined_insights}

---

Please use these insights to guide your modeling approach for the current problem.
"""
            else:
                # Fallback to original formatting if refinement fails
                memory_context = memory_bank.format_retrieved_cases_for_prompt(similar_cases)
        else:
            # Use original formatting (full cases)
            memory_context = memory_bank.format_retrieved_cases_for_prompt(similar_cases)
    
    # Build prompt with memory context
    prompt_template = get_prompt_template("default")
    system_prompt = prompt_template["system"]
    user_prompt = prompt_template["user"].format(question=problem_desc)
    
    # Inject memory context if available
    if memory_context:
        user_prompt = f"{memory_context}\n\n{user_prompt}"
    
    # Generate solution with self-healing retry mechanism
    full_prompt = f"{system_prompt}\n\n{user_prompt}"
    
    # Calculate prompt length for monitoring
    prompt_length = len(full_prompt)
    prompt_tokens_estimate = prompt_length // 4  # Rough estimate: 1 token ≈ 4 chars
    
    # Variables to track across attempts
    attempt_history = []
    final_response = ''
    final_code = ''
    execution_status = 'not_executed'
    execution_stdout = ''
    execution_stderr = ''
    execution_objective = None
    execution_returncode = None
    suggestions_path = ''
    executed_code_path = ''
    debug_signature = ''
    
    try:
        # Self-healing loop: try up to max_retries times
        current_prompt = full_prompt
        
        for attempt in range(1, max_retries + 1):
            print(f"  🔄 Attempt {attempt}/{max_retries} for problem {problem_id}")
            
            # Generate code
            response = get_response(current_prompt, model=model, temperature=temperature)
            code = extract_python_code(response)
            
            # Record this attempt
            attempt_info = {
                'attempt_number': attempt,
                'response': response,
                'code': code,
                'execution_status': 'not_executed',
            }

            if auto_debug and code.strip():
                target_dir = debug_output_dir or os.path.join(os.getcwd(), "auto_debug")
                os.makedirs(target_dir, exist_ok=True)
                
                # Execute the generated code
                exec_result = execute_generated_code(
                    code,
                    problem_id,
                    target_dir,
                    timeout=execution_timeout,
                )
                
                # Update attempt info
                attempt_info['execution_status'] = exec_result.status
                attempt_info['objective_value'] = exec_result.objective_value
                attempt_info['stdout'] = exec_result.stdout[:200] if exec_result.stdout else ''
                attempt_info['stderr'] = exec_result.stderr[:200] if exec_result.stderr else ''
                
                # Check if execution was successful
                if exec_result.status == 'success':
                    # Success! Use this result
                    print(f"  ✅ Success on attempt {attempt}")
                    execution_status = exec_result.status
                    execution_stdout = exec_result.stdout
                    execution_stderr = exec_result.stderr
                    execution_objective = exec_result.objective_value
                    execution_returncode = exec_result.returncode
                    executed_code_path = exec_result.code_path or ''
                    final_response = response
                    final_code = code
                    attempt_history.append(attempt_info)
                    break  # Exit the retry loop
                else:
                    # Failure - prepare for retry
                    print(f"  ❌ Failed on attempt {attempt}: {exec_result.status}")
                    execution_status = exec_result.status
                    execution_stdout = exec_result.stdout
                    execution_stderr = exec_result.stderr
                    execution_returncode = exec_result.returncode
                    executed_code_path = exec_result.code_path or ''
                    final_response = response
                    final_code = code
                    
                    # Write debug report
                    suggestions_path = write_debug_report(
                        problem_id,
                        problem_desc,
                        exec_result,
                        target_dir,
                    )
                    
                    # Record to debug store
                    error_message = execution_stderr or execution_stdout or execution_status
                    if debug_store:
                        debug_signature = debug_store.record_execution_feedback(
                            problem_id=problem_id,
                            description=problem_desc,
                            status=execution_status,
                            error_text=error_message,
                            guidance=f"Attempt {attempt}/{max_retries} failed. Review the debug report.",
                            source="generate_with_memory.auto_debug.self_healing",
                            metadata={
                                "attempt": attempt,
                                "returncode": execution_returncode,
                                "code_path": executed_code_path,
                            },
                        )
                    
                    attempt_history.append(attempt_info)
                    
                    # If not the last attempt, prepare retry prompt
                    if attempt < max_retries:
                        guidance_text = ""
                        if debug_case_bank and error_message:
                            debug_cases = debug_case_bank.retrieve_similar_cases(
                                error_message,
                                top_k=debug_case_top_k,
                            )
                            guidance_text = format_debug_cases_for_prompt(debug_cases)
                        error_feedback = build_error_feedback_prompt(
                            exec_result,
                            attempt,
                            code,
                            debug_guidance=guidance_text,
                        )
                        # Append error feedback to the prompt for next attempt
                        current_prompt = f"{full_prompt}\n\n{error_feedback}"
                        print(f"  🔧 Preparing retry with error feedback...")
                    else:
                        print(f"  ⚠️  Max retries ({max_retries}) reached, giving up")
            
            elif not code.strip():
                # No code generated
                attempt_info['execution_status'] = 'no_code'
                attempt_history.append(attempt_info)
                execution_status = 'no_code'
                execution_stderr = 'Generated code block is empty.'
                final_response = response
                final_code = code
                
                if attempt < max_retries:
                    # Retry with feedback about missing code
                    feedback = "\n\nYour previous response did not contain any Python code. Please provide the complete Gurobi code in a ```python``` code block."
                    current_prompt = f"{full_prompt}\n\n{feedback}"
                    print(f"  ⚠️  No code generated, retrying...")
                else:
                    print(f"  ⚠️  Max retries reached, no code generated")
                    break
            
            elif not auto_debug:
                # Auto debug disabled, just use the generated code
                execution_status = 'skipped'
                final_response = response
                final_code = code
                attempt_history.append(attempt_info)
                break

        if auto_debug:
            if execution_status == 'success':
                final_status = 'success'
            elif final_code.strip():
                final_status = 'execution_failed'
            else:
                final_status = 'no_code'
        else:
            final_status = 'success' if final_code.strip() else 'no_code'

        return {
            'id': problem_id,
            'model': model,
            'temperature': temperature,
            'description': problem_desc,
            'full_input_prompt': full_prompt,  # 💾 Complete input for reproducibility
            'refined_insights': refined_insights if use_llm_refinement else '',  # LLM-refined insights
            'prompt_length_chars': prompt_length,
            'prompt_length_tokens_est': prompt_tokens_estimate,
            'raw_response': final_response,
            'generated_code': final_code,
            'retrieved_cases': len(similar_cases),
            'original_retrieved': original_retrieved,
            'use_llm_refinement': use_llm_refinement,
            'status': final_status,
            'execution_status': execution_status,
            'execution_stdout': execution_stdout,
            'execution_stderr': execution_stderr,
            'execution_objective_value': execution_objective,
            'execution_returncode': execution_returncode,
            'debug_suggestions_path': suggestions_path,
            'executed_code_path': executed_code_path if executed_code_path else '',
            'debug_signature': debug_signature,
            'auto_debug_enabled': auto_debug,
            'execution_timeout_sec': execution_timeout if auto_debug else None,
            'max_retries': max_retries,
            'total_attempts': len(attempt_history),
            'attempt_history': attempt_history,
            'self_healing_enabled': True,
        }
    
    except Exception as e:
        print(f"Error generating solution for problem {problem_id}: {e}")
        
        # Still save the prompt even on error
        full_prompt = f"{system_prompt}\n\n{user_prompt}"
        
        return {
            'id': problem_id,
            'model': model,
            'temperature': temperature,
            'description': problem_desc,
            'full_input_prompt': full_prompt,  # Save even on error
            'refined_insights': '',
            'prompt_length_chars': len(full_prompt),
            'prompt_length_tokens_est': len(full_prompt) // 4,
            'raw_response': '',
            'generated_code': '',
            'retrieved_cases': len(similar_cases) if similar_cases else 0,
            'original_retrieved': original_retrieved,
            'use_llm_refinement': use_llm_refinement,
            'status': 'error',
            'error': str(e),
            'execution_status': 'not_executed',
            'execution_stdout': '',
            'execution_stderr': '',
            'execution_objective_value': None,
            'execution_returncode': None,
            'debug_suggestions_path': '',
            'executed_code_path': '',
            'debug_signature': '',
            'auto_debug_enabled': auto_debug,
            'execution_timeout_sec': execution_timeout if auto_debug else None,
            'max_retries': max_retries,
            'total_attempts': 0,
            'attempt_history': [],
            'self_healing_enabled': True,
        }


def generate_single_problem(
    problem: Dict,
    memory_bank: MemoryBank,
    model: str,
    temperature: float,
    top_k: int,
    filter_perfect: bool,
    use_llm_refinement: bool,
    *,
    auto_debug: bool,
    execution_timeout: int,
    debug_output_dir: Optional[str],
    debug_store: Optional[DebugMemoryStore],
    max_retries: int = 3,
    debug_case_bank: Optional[MemoryBank] = None,
    debug_case_top_k: int = 3,
) -> Dict:
    """
    Wrapper for parallel execution
    """
    problem_id = problem['id']
    problem_desc = problem['description']
    
    result = generate_with_memory(
        problem_id, problem_desc, memory_bank,
        model, temperature, top_k, filter_perfect, use_llm_refinement,
        auto_debug=auto_debug,
        execution_timeout=execution_timeout,
        debug_output_dir=debug_output_dir,
        debug_store=debug_store,
        max_retries=max_retries,
        debug_case_bank=debug_case_bank,
        debug_case_top_k=debug_case_top_k,
    )
    
    # Add ground truth
    result['answer'] = problem.get('answer', '')
    
    return result


def main():
    parser = argparse.ArgumentParser(description='Generate with Memory (parallel single solutions)')
    parser.add_argument('--dataset', type=str, default='IndustryOR',
                       help='Dataset name')
    parser.add_argument('--model', type=str, default='gpt-4o',
                       help='Model name')
    parser.add_argument('--temperature', type=float, default=0.01,
                       help='Temperature for generation')
    parser.add_argument('--max_problems', type=int, default=None,
                       help='Maximum number of problems to process')
    parser.add_argument('--output', type=str, required=True,
                       help='Output file path (JSONL)')
    parser.add_argument('--memory_dir', type=str, default=str(DEFAULT_MEMORY_DIR),
                       help='Memory storage directory')
    parser.add_argument('--embedding_model', type=str, default=None,
                       help='Optional embedding model name or local path for memory retrieval')
    parser.add_argument('--memory_top_k', type=int, default=4,
                       help='Number of cases to retrieve from memory (default: 4)')
    parser.add_argument('--no_filter_perfect', action='store_true',
                       help='Disable filtering of perfect similarity matches')
    parser.add_argument('--use_llm_refinement', action='store_true',
                       help='Use LLM to refine/summarize retrieved cases (improves quality, costs more API calls)')
    parser.add_argument('--parallel', type=int, default=5,
                       help='Number of parallel workers')
    parser.add_argument('--execution_timeout', type=int, default=120,
                       help='Timeout (seconds) for executing generated code during auto-debug')
    parser.add_argument('--no_auto_debug', action='store_true',
                       help='Disable automatic execution and debug capture for generated code')
    parser.add_argument('--debug_output_dir', type=str, default=None,
                       help='Directory to store auto-debug artifacts (code, logs, suggestions)')
    parser.add_argument('--debug_memory_path', type=str, default=str(DEFAULT_DEBUG_MEMORY),
                       help='Path to persistent debug memory JSONL file')
    parser.add_argument('--debug_case_memory_dir', type=str, default=str(DEFAULT_DEBUG_CASE_MEMORY),
                       help='Directory containing consolidated debug-case memory (built via build_debug_memory.py)')
    parser.add_argument('--debug_case_memory_top_k', type=int, default=3,
                       help='How many debug memory cases to retrieve when execution fails')
    parser.add_argument('--max_retries', type=int, default=3,
                       help='Maximum number of retry attempts for self-healing (default: 3)')
    
    args = parser.parse_args()

    args.dataset = normalize_dataset_name(args.dataset)

    auto_debug_enabled = not args.no_auto_debug
    debug_output_dir = args.debug_output_dir
    debug_store: Optional[DebugMemoryStore] = None
    if auto_debug_enabled:
        if debug_output_dir is None:
            base_dir = os.path.dirname(args.output) or '.'
            debug_output_dir = os.path.join(base_dir, 'auto_debug')
        os.makedirs(debug_output_dir, exist_ok=True)
        debug_store = DebugMemoryStore(args.debug_memory_path)
    else:
        debug_output_dir = None

    debug_case_bank: Optional[MemoryBank] = None
    if auto_debug_enabled and args.debug_case_memory_top_k > 0 and args.debug_case_memory_dir:
        case_dir = Path(args.debug_case_memory_dir)
        if case_dir.exists():
            try:
                if args.embedding_model:
                    debug_case_bank = MemoryBank(
                        memory_dir=str(case_dir),
                        embedding_model=args.embedding_model,
                    )
                else:
                    debug_case_bank = MemoryBank(memory_dir=str(case_dir))
            except Exception as exc:  # noqa: BLE001
                print(f"⚠️  Warning: failed to load debug-case memory from {case_dir} ({exc})")
        else:
            print(f"ℹ️  Debug-case memory directory not found: {case_dir} (skipping retrieval)")
    
    print("="*80)
    print("🧠 Generate with Memory (Parallel)")
    print("="*80)
    print(f"Dataset:      {args.dataset}")
    print(f"Model:        {args.model}")
    print(f"Temperature:  {args.temperature}")
    print(f"Memory dir:   {args.memory_dir}")
    if args.embedding_model:
        print(f"Embedding:    {args.embedding_model}")
    print(f"Memory Top-K: {args.memory_top_k}")
    print(f"Filter perfect matches: {not args.no_filter_perfect}")
    print(f"LLM Refinement: {'✅ Enabled' if args.use_llm_refinement else '❌ Disabled'}")
    print(f"Parallel:     {args.parallel}")
    print(f"Output:       {args.output}")
    print(f"Auto Debug:   {'✅ Enabled' if auto_debug_enabled else '❌ Disabled'}")
    if auto_debug_enabled:
        print(f"  Debug dir:      {debug_output_dir}")
        if args.debug_memory_path:
            print(f"  Debug memory:   {args.debug_memory_path}")
        print(f"  Exec timeout:   {args.execution_timeout}s")
        print(f"  Max retries:    {args.max_retries} (Self-healing enabled)")
    print("="*80)
    print()
    
    # Initialize memory bank only when retrieval is active.
    if args.memory_top_k > 0:
        print("Initializing memory bank...")
        if args.embedding_model:
            memory_bank = MemoryBank(memory_dir=args.memory_dir, embedding_model=args.embedding_model)
        else:
            memory_bank = MemoryBank(memory_dir=args.memory_dir)
        print()
    else:
        print("Skipping memory bank initialization because memory_top_k=0")
        print()
        memory_bank = NoOpMemoryBank()
    
    # Load dataset
    problems = load_dataset(args.dataset)
    if args.max_problems:
        problems = problems[:args.max_problems]
    
    print(f"Processing {len(problems)} problems with {args.parallel} workers")
    print()
    
    # Create output directory
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    
    # Parallel generation
    results = []
    
    if args.parallel <= 1:
        # Sequential processing
        for problem in tqdm(problems, desc="Generating"):
            result = generate_single_problem(
                problem, memory_bank, args.model, args.temperature,
                args.memory_top_k, not args.no_filter_perfect, args.use_llm_refinement,
                auto_debug=auto_debug_enabled,
                execution_timeout=args.execution_timeout,
                debug_output_dir=debug_output_dir,
                debug_store=debug_store,
                max_retries=args.max_retries,
                debug_case_bank=debug_case_bank,
                debug_case_top_k=args.debug_case_memory_top_k,
            )
            results.append(result)
    else:
        # Parallel processing
        with ThreadPoolExecutor(max_workers=args.parallel) as executor:
            futures = {
                executor.submit(
                    generate_single_problem,
                    problem, memory_bank, args.model, args.temperature,
                    args.memory_top_k, not args.no_filter_perfect, args.use_llm_refinement,
                    auto_debug=auto_debug_enabled,
                    execution_timeout=args.execution_timeout,
                    debug_output_dir=debug_output_dir,
                    debug_store=debug_store,
                    max_retries=args.max_retries,
                    debug_case_bank=debug_case_bank,
                    debug_case_top_k=args.debug_case_memory_top_k,
                ): problem for problem in problems
            }
            
            for future in tqdm(as_completed(futures), total=len(problems), desc="Generating"):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    problem = futures[future]
                    print(f"Error processing problem {problem['id']}: {e}")
    
    # Sort by problem ID
    results.sort(key=lambda x: x['id'])
    
    # Save results
    with open(args.output, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
    
    print()
    print("="*80)
    print("✅ Generation Complete")
    print("="*80)
    print(f"Total problems: {len(results)}")
    status_counts = Counter(r.get('status', 'unknown') for r in results)
    print(f"Successful:     {status_counts.get('success', 0)}")
    print(f"Errors:         {status_counts.get('error', 0)}")
    print(f"Results saved to: {args.output}")
    if status_counts:
        print("Status breakdown:")
        for status, count in sorted(status_counts.items()):
            print(f"  {status:<18}: {count}")
    
    # Memory statistics
    total_retrieved = sum(r.get('retrieved_cases', 0) for r in results)
    total_original = sum(r.get('original_retrieved', 0) for r in results)
    filtered = total_original - total_retrieved
    
    # Prompt length statistics
    prompt_lengths = [r.get('prompt_length_tokens_est', 0) for r in results if r.get('status') == 'success']
    avg_prompt_tokens = sum(prompt_lengths) / len(prompt_lengths) if prompt_lengths else 0
    max_prompt_tokens = max(prompt_lengths) if prompt_lengths else 0
    
    print()
    print("Memory Statistics:")
    print(f"  Total retrievals: {total_original}")
    print(f"  After filtering:  {total_retrieved}")
    print(f"  Filtered out:     {filtered} (perfect matches)")
    print(f"  Avg per problem:  {total_retrieved / len(results):.2f}")
    print()
    print("Prompt Length Statistics:")
    print(f"  Avg prompt tokens: {avg_prompt_tokens:.0f}")
    print(f"  Max prompt tokens: {max_prompt_tokens:.0f}")
    print(f"  ℹ️  All prompts saved in 'full_input_prompt' field")
    print("="*80)


if __name__ == "__main__":
    main()
