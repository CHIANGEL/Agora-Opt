import argparse
import json
import os
import signal
from main import chain_of_experts
from utils import extract_code_from_string


def CoE_solve(description_or_problem, model_name='gpt-4o', work_dir='.'):
    if isinstance(description_or_problem, dict):
        description = description_or_problem.get('description', '')
        code_example = description_or_problem.get('code_example', '')
    else:
        description = description_or_problem
        code_example = ''

    problem_data = {
        'description': description,
        'code_example': code_example
    }

    # Set API credentials from environment variables
    api_key = os.getenv("API_KEY")
    base_url = os.getenv("API_URL")

    full_response = chain_of_experts(
        problem_data,
        model_name=model_name,
        api_key=api_key,
        base_url=base_url,
        max_collaborate_nums=3,
        enable_reflection=True,
        max_trials=3,
        work_dir=work_dir)
    extracted_code = extract_code_from_string(full_response)
    return full_response, extracted_code

def handler(signum, frame):
        raise TimeoutError("Code execution exceeded the time limit.")


def run_code(code: str, timeout: int = 90) -> float:
    local_vars = {'__name__': '__main__'}
    
    # Register the timeout handler.
    signal.signal(signal.SIGALRM, handler)
    # Start the timeout alarm.
    signal.alarm(timeout)
    
    try:
        # Execute the generated code.
        exec(code, local_vars, local_vars)
        
        # Try an explicitly defined objVal first.
        objVal = local_vars.get('objVal')
        if objVal is not None:
            return float(objVal)
        
        # Otherwise try to recover the objective from a Gurobi model object.
        for var_name, var_val in local_vars.items():
            # Check whether the variable exposes a Gurobi-style objVal field.
            if hasattr(var_val, 'objVal'):
                return float(var_val.objVal)
                
    except TimeoutError as e:
        print(f"Execution timed out: {e}")
    except Exception as e:
        print(f"Error during code execution: {e}")
    finally:
        # Always clear the alarm.
        signal.alarm(0)
        
    return None
        
        
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, help='Path to the dataset file')
    parser.add_argument('--model', type=str, default='gpt-4o', help='Model to use for generation')
    parser.add_argument('--output-dir', type=str, required=True, help='Directory to save the results')
    args = parser.parse_args()

    with open(args.dataset, "r") as f:
        lines = f.readlines()
        
    benchmark_name = os.path.splitext(os.path.basename(args.dataset))[0]
    
    length_lines = len(lines)
    for id_count in range(length_lines):
        # Register the timeout handler.
        signal.signal(signal.SIGALRM, handler)
        # Start the timeout alarm.
        signal.alarm(600)
        try:
            print(f"\n\n\n\n\nProcessing problem_{id_count}...")
            
            result_file_dir = os.path.join(args.output_dir, args.model, benchmark_name, f"problem_{id_count}")
            os.makedirs(result_file_dir, exist_ok=True)

            # Skip problems that already have code.py.
            if os.path.exists(os.path.join(result_file_dir, 'code.py')):
                print(f"code.py already exists in {result_file_dir}, skipping...")
                continue
            
            data = json.loads(lines[id_count])
            
            description = data["description"]
            answer = data["answer"]
            
            print(f"Processing {benchmark_name} problem_{id_count}...")
            
            # Generate code with the CoE solver.
            problem_data = {
                'description': description,
                'code_example': ''
            }

            try:
                full_response, extracted_code = CoE_solve(problem_data, model_name=args.model, work_dir=result_file_dir)
            except Exception as e:
                print(f"Error during CoE_solve: {e}")
                continue
            
            
            with open(os.path.join(result_file_dir, 'response.txt'), 'w') as rf:
                rf.write(full_response)
                
            with open(os.path.join(result_file_dir, 'code.py'), 'w') as cf:
                cf.write(extracted_code)

            out_path = os.path.join(result_file_dir, 'meta.json')
            meta = {
                'description': description,
                'ground_truth': answer,
                'generated_raw': full_response,
                'extracted_code': extracted_code
            }

            meta['objVal'] = None

            # Save metadata.
            with open(out_path, 'w') as wf:
                json.dump(meta, wf, indent=2)
        except TimeoutError as e:
            print(f"Processing problem_{id_count} timed out: {e}")
        except Exception as e:
            print(f"Error processing problem_{id_count}: {e}")
        finally:
            # Always clear the alarm.
            signal.alarm(0)
        
    # Analyze evaluation results.
    count = 0
    correct = 0
    results_dir = os.path.join(args.output_dir, args.model, benchmark_name)
    result_files = os.listdir(results_dir)
    
    for rf in result_files:
        problem_dir = os.path.join(results_dir, rf)
        # Execute code.py once more to recover objVal.
        code_path = os.path.join(problem_dir, 'code.py')
        if not os.path.exists(code_path):
            continue

        with open(code_path, 'r') as cf:
            code = cf.read()
            objVal = run_code(code)
            print(f"Problem {rf}, objVal: {objVal}")
            
        # Overwrite objVal in meta.json.
        meta_path = os.path.join(problem_dir, 'meta.json')
        with open(meta_path, 'r') as mf:
            meta = json.load(mf)
            meta['objVal'] = objVal
        with open(meta_path, 'w') as mf:
            json.dump(meta, mf, indent=2)
        
        
        with open(meta_path, 'r') as mf:
            meta = json.load(mf)
            ground_truth = meta.get('ground_truth', None)
            objVal = meta.get('objVal', None)
            
            if ground_truth is not None and objVal is not None and ground_truth != 'No Best Solution':
                try:
                    ground_truth = float(ground_truth)
                    objVal = float(objVal)
                    diff = abs(ground_truth - objVal)
                    if diff <= 0.01:
                        correct += 1
                
                except (TypeError, ValueError):
                    pass
                
            count += 1
        
    print(f"Correct: {correct}, Total: {count}")
    print(f"Accuracy: {correct/count:.2%}" if count > 0 else "Accuracy: N/A (no samples)")
