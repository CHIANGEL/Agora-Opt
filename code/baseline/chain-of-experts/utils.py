import re
import json
import ast
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


def _unwrap_code_fence(text):
    if not isinstance(text, str):
        return text
    fenced = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE)
    if fenced:
        return fenced.group(1).strip()
    return text.strip()


def _candidate_json_snippets(text):
    candidates = []
    raw = text.strip()
    if raw:
        candidates.append(raw)

    unwrapped = _unwrap_code_fence(raw)
    if unwrapped and unwrapped not in candidates:
        candidates.append(unwrapped)

    first_obj = unwrapped.find('{')
    last_obj = unwrapped.rfind('}')
    if 0 <= first_obj < last_obj:
        obj_snippet = unwrapped[first_obj:last_obj + 1]
        if obj_snippet not in candidates:
            candidates.append(obj_snippet)

    first_arr = unwrapped.find('[')
    last_arr = unwrapped.rfind(']')
    if 0 <= first_arr < last_arr:
        arr_snippet = unwrapped[first_arr:last_arr + 1]
        if arr_snippet not in candidates:
            candidates.append(arr_snippet)

    return candidates


def safe_json_loads(raw_text, default=None):
    if raw_text is None:
        return default

    text = str(raw_text).strip().strip("'").strip()
    if not text:
        return default

    for candidate in _candidate_json_snippets(text):
        try:
            return json.loads(candidate)
        except (json.JSONDecodeError, TypeError):
            pass

        try:
            parsed = ast.literal_eval(candidate)
            if isinstance(parsed, (dict, list)):
                return parsed
        except (SyntaxError, ValueError):
            pass

        fixed = candidate.replace("'''", '"').replace("'", '"')
        try:
            return json.loads(fixed)
        except (json.JSONDecodeError, TypeError):
            pass

    return default



def read_problem(dataset, problem_name):
    base_dir = 'dataset'
    with open(os.path.join(base_dir, dataset, problem_name, 'description.txt'), 'r', encoding='utf8') as f:
        description = f.read()

    return {
        'description': description,
        'code_example': ''
    }
