import os
import json
import numpy as np
from comment import Comment
from conductor import Conductor
from reducer import Reducer
from evaluator import Evaluator
from experts import (
    ModelingExpert, 
    ProgrammingExpert,
    LPFileGenerator,
    ModelingKnowledgeSupplementExpert,
    ParameterExtractor,
    CodeReviewer,
    ProgrammingExampleProvider,
    TerminologyInterpreter,
)
from comment_pool import CommentPool
from utils import extract_code_from_string, safe_json_loads


def chain_of_experts(problem,
                    max_collaborate_nums,
                    model_name,
                    api_key,
                    base_url,
                    enable_reflection,
                    max_trials,
                    work_dir='.'):
    """Run Chain of Experts pipeline

    Args:
        problem: a dict of problem_description and code_example.
        work_dir: directory to write generated_code.py for evaluation.

    Return:
        code: code of problem
    """
    all_experts = [
        TerminologyInterpreter(model_name, api_key, base_url),
        ParameterExtractor(model_name, api_key, base_url),
        ModelingExpert(model_name, api_key, base_url),
        ProgrammingExampleProvider(model_name, api_key, base_url),
        ProgrammingExpert(model_name, api_key, base_url),
        # LPFileGenerator(model_name),
        ModelingKnowledgeSupplementExpert(model_name, api_key, base_url),
        CodeReviewer(model_name, api_key, base_url),
    ]
    num_experts = len(all_experts)
    reducer = Reducer(model_name, api_key, base_url)
    comment_pool = CommentPool(all_experts, visible_matrix=np.ones((num_experts, num_experts)))
    conductor = Conductor(model_name, api_key, base_url)
    evaluator = Evaluator(model_name, api_key, base_url)
    expert_stack = []

    for _ in range(max_trials):
        for _ in range(max_collaborate_nums):
            next_expert = conductor.forward(problem, comment_pool, max_collaborate_nums)
            print(f'Choose next expert: {next_expert.name}')
            comment_text = next_expert.forward(problem, comment_pool)
            print(f'Given comment:\n{comment_text}')
            comment_pool.add_comment(Comment(next_expert, comment_text))
            expert_stack.append(next_expert)
        answer = reducer.forward(problem, comment_pool)

        code = extract_code_from_string(answer)
        code_path = os.path.join(work_dir, 'generated_code.py')
        os.makedirs(work_dir, exist_ok=True)
        with open(code_path, 'w') as f:
            f.write(code)

        if enable_reflection:
            test_sample = evaluator.forward(problem)
            print(f'Generate test sample:\n{test_sample}')
            test_samples = [test_sample]
            feedback = evaluator.evaluate(test_samples, code_path=code_path)
            feedback_pool = CommentPool(all_experts, visible_matrix=np.ones((num_experts, num_experts)))
            feedback_pool.add_comment(Comment(evaluator, feedback))
            if feedback is not None:
                while expert_stack:
                    previous_expert = expert_stack.pop()
                    previous_comment = comment_pool.pop_comment()
                    result = previous_expert.backward(feedback_pool)
                    result_json = safe_json_loads(
                        result,
                        default={"is_caused_by_you": False, "reason": str(result), "refined_result": previous_comment.comment_text}
                    )

                    if not isinstance(result_json, dict):
                        result_json = {"is_caused_by_you": False, "reason": str(result), "refined_result": previous_comment.comment_text}

                    result_json.setdefault("is_caused_by_you", False)
                    result_json.setdefault("reason", "")
                    result_json.setdefault("refined_result", previous_comment.comment_text)

                    print("result:", result_json)
                    result = result_json
                    if result["is_caused_by_you"]:
                        previous_comment.comment_text = result["refined_result"]
                        expert_stack.append(previous_expert)
                        comment_pool.add_comment(previous_comment)
                        break
                    else:
                        feedback_pool.add_comment(Comment(previous_expert, result['reason']))
            else:
                break
    return answer


if __name__ == '__main__':
    from utils import read_problem
    problem = read_problem('LPWP', 'prob_250')
    chain_of_experts(problem, model_name='gpt-3.5-turbo-1106', enable_reflection=False)
