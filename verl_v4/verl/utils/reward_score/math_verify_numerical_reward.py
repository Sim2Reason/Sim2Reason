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
        correct = basic_utils.math_verify_numerical_answer(str(solution_str), str(ground_truth), max_relative_error=0.01)

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
        "parsed_answer": str(string_in_last_boxed), # this has to be str
    }
    return results

if __name__ == "__main__":
    print(compute_score("The answer is $999$", "1000"))


    cases = [
        # ground truth, prediction, expected
        ("2.5", "2.5", True),
        ("2.499", "2.5", True),
        ("2.47",  "2.5", True),
        ("2.44",  "2.5", False),
        ("2.44",  "2.4", True),
        ("123000", "123000", True),
        ("120000", "123000", True),
        ("100000", "120000", False),    # 如果gt是没有小数，pred_answer的十分位上的不允许四舍五入到个位再比较，不然进度太低了
        ("110000", "120000", False),
        ("0.000457", "0.000456", False),
        ("0.000457", "0.00046", True),
        ("0.000457", "0.00045", False),
        ("0.000457", "0.0004573", True),
        ("0.000457", "0.0004575", False),
        ("-999.9", "-1000", True),
        ("0", "0", True),
        ("0", "0.01", False),
        ("0", "0.0001", False),
        ("1000", "1000.005", True),
        ("1000", "1000.5", False),
        ("1000", "1000.1", True),
        ("432.23", "432.219999", False),
        ("432.23", "432.229", True),
        ("0.001", "0.0012", False),
    ]

    for pred, gt, expected in cases:
        print(f"Pred: {pred}, GT: {gt}, Expected: {expected}")
        print(basic_utils.math_verify_numerical_answer(pred, gt))
        assert basic_utils.math_verify_numerical_answer(pred, gt) == expected
        print("-" * 50)