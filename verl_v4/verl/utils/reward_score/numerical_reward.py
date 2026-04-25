import sys
import os
import ipdb

def find_project_root(target_dir_name="Sim2Reason"):
    path = os.path.abspath(__file__)
    print(f"Current file path: {path}")
    while True:
        path = os.path.dirname(path)
        if os.path.basename(path) == target_dir_name:
            return path
        if path == "/" or path == "":
            raise FileNotFoundError(f"Could not find directory '{target_dir_name}' in path hierarchy.")

Sim2Reason_path = find_project_root("Sim2Reason")
print(f"Sim2Reason path: {Sim2Reason_path}")

sys.path.append(Sim2Reason_path)


from llm.utils.math_utils import extract_answer
import llm.utils.basic_utils as basic_utils

import re

st = ipdb.set_trace

def compute_score(solution_str, ground_truth) -> dict:
    retval = 0.
    has_boxed = 0
    parse_error = 0
    string_in_last_boxed = ""
    # st()
    try:
        # for debugging
        boxed_match = re.findall(r'\\boxed{(.+)}', solution_str, re.DOTALL)
        if boxed_match:
            has_boxed = 1
        else:
            has_boxed = 0
        
        # for reward
        string_in_last_boxed = extract_answer(solution_str, 'physics')
        correct, verify_reason = basic_utils.check_answer_heuristic(string_in_last_boxed, str(ground_truth), 2, "exact")
        if correct:
            retval = 1.
        
    except Exception as e:
        parse_error = 1
        string_in_last_boxed = e   # to print the error
        
        # print(e)
    results = {
        "score": retval,
        "has_boxed": has_boxed,
        "parse_error": parse_error,
        "parsed_answer": str(string_in_last_boxed),
    }
    return results

