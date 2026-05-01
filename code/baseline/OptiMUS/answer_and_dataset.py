"""Dataset loading, preprocessing, answer extraction, and validation helpers."""
import os
import json
import math
import re


SUPPORTED_DATASETS = ["ComplexLP", "EasyLP", "IndustryOR", "NL4Opt", "NLP4LP", "ReSocratic", "ComplexOR", "IndustryOR_v2"]
CWD = os.getcwd()


def get_desc_and_answer(dataset):
    '''
    Input: dataset name, e.g., "ComplexLP"
    Output: list of (description, answer) tuples
    '''
    dataset_path = f"{CWD}/clean_benchmarks/{dataset}_clean.jsonl"
    # assert dataset in SUPPORTED_DATASETS, f"Unsupported dataset: {dataset}. Supported datasets are: {SUPPORTED_DATASETS}"
    assert os.path.exists(dataset_path), f"Dataset file not found: {dataset_path}"
    
    data = []
    with open(dataset_path, "r") as f:
        lines = f.readlines()
        for line in lines:
            try:
                obj = json.loads(line.strip())
                description = obj.get("description", "")
                answer = obj.get("answer", "{}")
                data.append((description, answer))
            except Exception as e:
                print(f"Error parsing line: {line}. Error: {e}")

    return data


def get_desc_and_answer_for(dataset, prob_id):
    '''
    Input: dataset name, e.g., "ComplexLP", prob_id (0-indexed), id of the clean_benchmarks/{dataset}_clean.jsonl file
    Output: (description, answer) tuple for the given prob_id
    '''
    data = get_desc_and_answer(dataset)
    assert 0 <= prob_id < len(data), f"Invalid prob_id: {prob_id}. It should be between 0 and {len(data)-1}."
    return data[prob_id]
    

def get_answer_from_output(output: str) -> float:
    '''
    Extract the optimal objective value from Gurobi output.
    Input: output string from Gurobi
    Output: float value of the optimal objective
    '''
    if output is None or output.strip() == "":
        print("No output provided.")
        return None
    
    lines = output.split("\n")
    answer = None
    for line in lines:
        if "Optimal Objective Value:" in line:
            answer = line.split(":")[-1].strip()
        elif "Optimal Objective Value (Total Machines):" in line:
            answer = line.split(":")[-1].strip()
        elif "Optimal objective:" in line:
            # Keep only the numeric part.
            match = re.search(r"Optimal objective:\s*([-\d.]+)", line)
            if match:
                answer = match.group(1)
        elif "Optimal objective" in line:
            match = re.search(r"Optimal objective\s*[:=]?\s*([-]?\d*\.?\d+([eE][-+]?\d+)?)", line)
            if match:
                answer = match.group(1)
        elif "Best objective" in line:
            # Extract the numeric value after "Best objective".
            match = re.search(r"Best objective\s*[:=]?\s*([-]?\d*\.?\d+([eE][-+]?\d+)?)", line)
            if match:
                answer = match.group(1)
            
    if answer is None:
        # Handle other outcomes such as infeasible runs.
        return None
    # Keep only numeric characters.
    answer = re.sub(r"[^\d.-]", "", answer)
    if answer == "":
        print("No valid answer found in the output.")
        return None
    else:
        try:
            return float(answer)
        except ValueError:
            print(f"Could not convert answer '{answer}' to float.")
            return None


def converge(answerA, answerB, delta=0.1, relative_delta = None):
    if relative_delta is None:
        if answerA is None or answerB is None:
            return False
        return math.fabs(float(answerA) - float(answerB)) < delta
    else:
        if answerA is None or answerB is None:
            return False
        return math.fabs(float(answerA) - float(answerB)) < relative_delta * (math.fabs(float(answerA))+1)


def check_answer(dataset_file):
    data = get_desc_and_answer(dataset_file)
    count = 0
    correct = 0
    for idx, (desc, ans) in enumerate(data):
        try:
            with open (os.path.join(dataset_file, f"problem_{idx}/output_solution.txt"), "r") as f:
                my_answer = float(f.read().strip())
            
            count += 1
            if converge(my_answer, ans):
                correct += 1
                print(f"Problem {idx} correct: got {my_answer}, expected {ans}")
            else:
                print(f"Problem {idx} incorrect: got {my_answer}, expected {ans}")
        except Exception as e:
            print(f"Problem {idx} error: {e} \n\nThe answer should be {ans}")
    print(f"Accuracy: {correct}/{count} = {correct/count*100:.2f}%")
