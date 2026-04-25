import ipdb
import json
import re
import numpy as np
import traceback
import polars as pl
import os
import sys
from importlib.metadata import version, PackageNotFoundError


def find_project_root(target_dir_name="Sim2Reason"):
    path = os.path.abspath(__file__)
    while True:
        path = os.path.dirname(path)
        if os.path.basename(path) == target_dir_name:
            return path
        if path == "/" or path == "":
            raise FileNotFoundError(f"Could not find directory '{target_dir_name}' in path hierarchy.")

Sim2Reason_path = find_project_root("Sim2Reason")
print(f"Sim2Reason path: {Sim2Reason_path}")

sys.path.append(Sim2Reason_path)


st = ipdb.set_trace
from llm.utils import basic_utils # TODO: use this to reduce redundancy

def compute_score(solution_str, ground_truth, extra_info):
    """
    Verifies the ground truth against the found solution for problems that
    have an equation as the answer.
    """
    assert version("antlr4-python3-runtime") == "4.11.0"
    parse_error = 0
    has_boxed = 0
    parsed_eq = ""
    given_variable_mapping_str = ""
    generated_answer = ""
    
    try:
        given_variable_mapping_str = extra_info['given_variable_mapping']
        given_variable_mapping = json.loads(given_variable_mapping_str)
        given_variable_mapping = {k:str(v).split(" ")[0] for k,v in given_variable_mapping.items()}
        given_variable_mapping.update({"g": 9.81})

        # Convert given variables to float
        substitution_dict = {}
        for k, v in given_variable_mapping.items():
            substitution_dict[k] = float(v)

        # for debugging
        boxed_match = re.findall(r'\\boxed{(.+)}', solution_str, re.DOTALL)
        if boxed_match:
            has_boxed = 1
        else:
            has_boxed = 0

        # for reward
        parsed_eq = basic_utils.extract_final_symbolic_answer(solution_str, not_equation = True)
        
        latex_compliant_substitution_dict = basic_utils.convert_sym_mapping_to_latex(substitution_dict)
        final_substitution_dict = {k: (v if "theta" not in str(k) else v / 180 * np.pi) for k, v in latex_compliant_substitution_dict.items()}
        # Substitute numeric values into the symbolic expression
        generated_answer = str(parsed_eq.subs(final_substitution_dict).evalf())
        
        correct, verify_reason = basic_utils.check_answer_heuristic(generated_answer, str(ground_truth), 2, "legacy")
        if correct:
            retval = 1.
        else:
            retval = 0.
    except Exception as e:
        # print("An error occurred:", e)
        full_error = traceback.format_exc()  # Capture the full stack trace as a string
        retval = 0.
        parse_error = 1
        parsed_eq = full_error  # Assign the full error message to parsed_eq

    
    results = {
        "score": retval,
        "has_boxed": has_boxed,
        "parse_error": parse_error,
        "parsed_answer": "parsed_eq: " + str(parsed_eq) + "\n\npred_answer: " + generated_answer + "\n\ngiven_variable_mapping: " + given_variable_mapping_str,
    }
    return results

def compute_score_updated(solution_str, ground_truth, extra_info):
    """
    Verifies the ground truth against the found solution for problems that
    have an letter (for MC answers), numerical, or equation as the answer.
    ground_truth is the letter, numerical, or equation.

    Args:
        solution_str (str): The solution string from the LLM.
        ground_truth (str): The ground truth string.
        extra_info (dict): The extra info dictionary.

    Returns:
        dict: A dictionary containing the score (0 or 1), has_boxed, parse_error, and parsed_answer.
    """
    pass


if __name__ == "__main__":
    compute_score_updated("solution_str", "ground_truth", "extra_info")
    df = pl.read_parquet("/home/mprabhud/datasets/physics_sim_data/v10/train_new_v10.parquet")
    df = df.filter(pl.col("data_source") == "math_p_symbolic")
    
    first_row = df.head(1)
    ground_truth = first_row["reward_model"][0]["ground_truth"]
    solution_str = r'''To solve for the kinetic energy of the 3rd entity after t seconds, we will utilize the provided data and the kinematic equation:

v = u + at

Given that the initial speed (u) is 2.1 m/s and the acceleration due to gravity (a) is 9.81 m/s^2, we can calculate the final velocity (v) after t seconds:.

v = 2.1 + (9.81 * t) .

The kinetic energy is calculated as:

K = (1/2) * m * v^2 + (1/2) * m * a * t^2

Substituting the values:

K = (1/2) * m_3 * (2.1 + 9.81 * t)^2 + (1/2) * m_3 * 9.81 * t^2

K = 0.5 * m_3 * (2.1 + 9.81 * t)^2 + 0.5 * m_3 * 9.81 * t^2

Therefore, the kinetic energy of the 3rd entity after t seconds is:.

\boxed{K = 0.5 * m_3 * [ (2.1 + 9.81 * t)^2 + 9.81 * t^2 ]}'''

    extra_info = first_row['extra_info'][0]
    extra_info["given_variable_mapping"] =  r'''{"m_1": "2.976", "m_2": "4.681", "m_3": "3.684", "m_4": "2.237", "t": 1.2189999999999765, "g": "9.81"}'''
    print()
    score = compute_score(solution_str, ground_truth, extra_info)
    print(score)
