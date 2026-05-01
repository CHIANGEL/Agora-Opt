import re
import json
import os


def extract_code_from_string(input_string):
    # Match code within ```python ... ```
    pattern = r'```python(.*?)```'
    
    # Find all matches in the input string
    code_blocks = re.findall(pattern, input_string, re.DOTALL)

    if len(code_blocks) == 0:
        # print(f'Parse code error! {input_string}')
        return input_string
    elif len(code_blocks) == 1:
        return code_blocks[0]


def read_problem(dataset, problem_name):
    base_dir = 'dataset'
    with open(os.path.join(base_dir, dataset, problem_name, 'description.txt'), 'r', encoding='utf8') as f:
        description = f.read()

    with open(os.path.join(base_dir, dataset, problem_name, 'code_example.py'), 'r', encoding='utf8') as f:
        code_example = f.read()

    return {
        'description': description,
        'code_example': code_example
    }
