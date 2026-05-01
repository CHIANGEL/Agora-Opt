import os
import json
import signal
import io
import re
import contextlib
from openai import OpenAI
from utils import extract_code_from_string


REL_TOL = 0.05
ABS_TOL_ZERO_GT = 1e-3
EVAL_TIMEOUT_SEC = 90

# Reuse the provided solve function
def solve(problem_data, model_name):
    problem_description = problem_data['description']
    prompt = f"""You are a Python programmer in the field of operations research and optimization. 
    Your proficiency in utilizing third-party libraries such as Gurobi is essential. In addition to your expertise in Gurobi, it would be great if you could also provide some background in related libraries or tools, like NumPy, SciPy, or PuLP.
    You are given a specific problem. You aim to develop an efficient Python program that addresses the given problem.
    Now the origin problem is as follow:\n{problem_description}
    Let's analyse the problem STEP BY STEP, but do not think for too long! Give your Python code in the end, with ```python ... ``` format."""

    client = OpenAI(
        api_key=os.getenv("API_KEY"),
        base_url=os.getenv("API_URL"),
    )
    
    resp = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": "You are a helpful assistant with chain of thought ability."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.01,
        max_tokens=40000,
        timeout=600,
    )
    answer = resp.choices[0].message.content
    code = extract_code_from_string(answer)
    return answer, code

def _timeout_handler(signum, frame):
    raise TimeoutError("execution timed out")


def _to_float(value):
    try:
        if isinstance(value, str):
            value = value.replace(",", "").replace("$", "").strip()
        return float(value)
    except Exception:
        return None


def _extract_obj_from_namespace(namespace):
    for key in ("objVal", "ObjVal", "objective", "objective_value"):
        if key in namespace:
            parsed = _to_float(namespace[key])
            if parsed is not None:
                return parsed

    for _, var_val in namespace.items():
        for attr in ("objVal", "ObjVal"):
            if hasattr(var_val, attr):
                parsed = _to_float(getattr(var_val, attr))
                if parsed is not None:
                    return parsed

    return None


def _extract_obj_from_output(output_text):
    patterns = [
        r"Optimal objective\s+([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)",
        r"Minimum Total Expenditure:\s*\$?\s*([+-]?\d[\d,]*(?:\.\d+)?)",
        r"(?:Objective|Total cost|Total Cost|objVal)\s*[:=]\s*\$?\s*([+-]?\d[\d,]*(?:\.\d+)?)",
    ]

    for pattern in patterns:
        matches = re.findall(pattern, output_text, flags=re.IGNORECASE)
        if matches:
            parsed = _to_float(matches[-1])
            if parsed is not None:
                return parsed

    return None


def run_code(code: str, code_path: str = None, context_label: str = None, timeout_sec: int = EVAL_TIMEOUT_SEC) -> float:
    namespace = {"__name__": "__main__"}
    original_cwd = os.getcwd()
    location = context_label or code_path or "<in-memory-code>"
    if code_path:
        namespace["__file__"] = os.path.abspath(code_path)

    old_handler = None
    try:
        # Mimic the `python code.py` execution environment as closely as possible.
        if code_path:
            os.chdir(os.path.dirname(os.path.abspath(code_path)))

        if hasattr(signal, "SIGALRM") and timeout_sec is not None and timeout_sec > 0:
            old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
            signal.alarm(int(timeout_sec))

        # Use a shared namespace so functions can access top-level imports and variables.
        output_buffer = io.StringIO()

        class _TeeStream:
            def __init__(self, *streams):
                self.streams = streams

            def write(self, data):
                for stream in self.streams:
                    stream.write(data)
                return len(data)

            def flush(self):
                for stream in self.streams:
                    stream.flush()

        tee_stdout = _TeeStream(output_buffer, os.sys.stdout)
        with contextlib.redirect_stdout(tee_stdout):
            exec(code, namespace, namespace)
        
        # Prefer explicit values from the namespace, then fall back to stdout parsing.
        obj_val = _extract_obj_from_namespace(namespace)
        if obj_val is not None:
            return obj_val
        obj_val = _extract_obj_from_output(output_buffer.getvalue())
        if obj_val is not None:
            return obj_val
        
        return None
    except TimeoutError:
        print(f"Error during code execution [{location}]: timeout after {timeout_sec}s")
        return None
    except Exception as e:
        print(f"Error during code execution [{location}]: {e}")
        return None
    finally:
        if hasattr(signal, "SIGALRM") and timeout_sec is not None and timeout_sec > 0:
            signal.alarm(0)
            if old_handler is not None:
                signal.signal(signal.SIGALRM, old_handler)
        os.chdir(original_cwd)
        
        
        

def run_benchmark_test(dataset_path, model_name, output_dir):
    # Part 1: Run experiments and generate code
    if not os.path.exists(dataset_path):
        print(f"Dataset file not found: {dataset_path}")
        return

    with open(dataset_path, "r") as f:
        lines = f.readlines()

    benchmark_name = os.path.splitext(os.path.basename(dataset_path))[0]

    for id_count, line in enumerate(lines):
        data = json.loads(line)

        description = data["description"]
        answer = data["answer"]

        result_file_dir = os.path.join(output_dir, model_name, benchmark_name, f"problem_{id_count}")
        os.makedirs(result_file_dir, exist_ok=True)
        print(f"Processing {benchmark_name} problem_{id_count}...")
        
        meta_path = os.path.join(result_file_dir, 'meta.json')
        if os.path.exists(meta_path):
            print(f"meta.json already exists in {result_file_dir}, skipping...")
            continue

        # Generate code with the solve function.
        problem_data = {
            'description': description,
        }
        full_response, extracted_code = solve(problem_data, model_name)

        with open(os.path.join(result_file_dir, 'response.txt'), 'w') as rf:
            rf.write(full_response)

        if extracted_code:
            with open(os.path.join(result_file_dir, 'code.py'), 'w') as cf:
                cf.write(extracted_code)

        meta = {
            'description': description,
            'ground_truth': answer,
            'generated_raw': full_response,
            'extracted_code': extracted_code
        }

        # Try executing the extracted code.
        if extracted_code:
            obj = run_code(extracted_code, code_path=os.path.join(result_file_dir, 'code.py'))
            meta['objVal'] = obj
        else:
            meta['objVal'] = None

        # Save metadata.
        with open(meta_path, 'w') as wf:
            json.dump(meta, wf, indent=2)

    # Part 2: Test result analysis
    count = 0
    correct = 0
    results_dir = os.path.join(output_dir, model_name, benchmark_name)
    if not os.path.exists(results_dir):
        print(f"Results directory not found: {results_dir}")
        return
        
    result_files = [d for d in os.listdir(results_dir) if os.path.isdir(os.path.join(results_dir, d))]

    for rf in result_files:
        problem_dir = os.path.join(results_dir, rf)
        code_path = os.path.join(problem_dir, 'code.py')
        meta_path = os.path.join(problem_dir, 'meta.json')

        if not os.path.exists(code_path) or not os.path.exists(meta_path):
            continue

        # Execute code.py once more to recover objVal.
        with open(code_path, 'r') as cf:
            code = cf.read()
            objVal = run_code(code, code_path=code_path, context_label=rf, timeout_sec=EVAL_TIMEOUT_SEC)
            print(f"Problem {rf}, objVal: {objVal}")

        # Overwrite objVal in meta.json.
        with open(meta_path, 'r') as mf:
            meta = json.load(mf)
            meta['objVal'] = objVal
        with open(meta_path, 'w') as mf:
            json.dump(meta, mf, indent=2)

        ground_truth = meta.get('ground_truth', None)
        objVal = meta.get('objVal', None)

        if ground_truth is not None and objVal is not None and ground_truth != 'No Best Solution':
            try:
                ground_truth_float = float(ground_truth)
                objVal_float = float(objVal)
                diff = abs(ground_truth_float - objVal_float)
                if ground_truth_float == 0:
                    is_correct = diff <= ABS_TOL_ZERO_GT
                else:
                    rel_error = diff / abs(ground_truth_float)
                    is_correct = rel_error <= REL_TOL

                if is_correct:
                    correct += 1
            except (TypeError, ValueError):
                pass
        
        count += 1

    if count > 0:
        print(f"Correct: {correct}, Total: {count}")
        print(f"Accuracy: {correct/count:.2%}")
    else:
        print("No results found to analyze.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run Chain-of-Thought benchmark test.")
    parser.add_argument('--dataset', type=str, required=True, help='Path to the dataset file (e.g. data/IndustryOR_clean.jsonl)')
    parser.add_argument('--model', type=str, required=True, help='Model name')
    parser.add_argument('--output-dir', type=str, required=True, help='Directory to save results')
    args = parser.parse_args()
    run_benchmark_test(dataset_path=args.dataset, model_name=args.model, output_dir=args.output_dir)
