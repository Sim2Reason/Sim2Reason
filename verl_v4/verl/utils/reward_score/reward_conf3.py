import ipdb
import json
import re
import numpy as np
import traceback
import polars as pl
import os
import sys
from importlib.metadata import version, PackageNotFoundError

## Reward Configuration:
# use_relative_precision = True
# max_relative_error=1e-2 
# mode='relative'
# relative_to_absolute_threshold=1e-30

class FallbackConfig:
    debug_print=False
    conservative_reward=False

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

from llm.utils import basic_utils
st = ipdb.set_trace

def compute_score(solution_str, ground_truth, data_source, extra_info, config = None):
    """
    Verifies the ground truth against the found solution for problems that
    have an equation as the answer.
    """

    if config is None: config = FallbackConfig
    
    boxed_list = basic_utils.extract_bracket_content(solution_str)

    has_boxed = len(boxed_list) > 0
    if "validation" in data_source or "hcv" in data_source or "ipho" in data_source or "Types_1-5_Full_Dataset" in data_source:
        try:
            correct, parsed_sol = basic_utils.math_verify_numerical_answer_simple_relative(solution_str, ground_truth, use_relative_precision=True, max_relative_error=1e-2, mode='relative', relative_to_absolute_threshold=1e-30, data_source=data_source, force_config_for_validation=False, config = config)
            parsed_sol = parsed_sol[-1]
            parse_error = 0
        except Exception as e:
            correct = False
            full_error = traceback.format_exc()  # Capture the full stack trace as a string
            if isinstance(e, IndexError) or isinstance(e, TypeError):
                parsed_sol = f"parsed_sol = {parsed_sol} \n\n IndexError: {str(e)}"
            else:
                parsed_sol = full_error
            parse_error = 1

    else: # Synthetic numerical or symbolic
        if "symbolic" in data_source:
            # For symbolic problems, we need to parse the solution string
            try:
                given_variable_mapping_str = extra_info['given_variable_mapping']
                given_variable_mapping = json.loads(given_variable_mapping_str)
                given_variable_mapping = {k:str(v).split(" ")[0] for k,v in given_variable_mapping.items()}
                given_variable_mapping.update({"g": 9.81})

                # Convert given variables to float
                substitution_dict = {}
                for k, v in given_variable_mapping.items():
                    substitution_dict[k] = float(v)

                parsed_exp = basic_utils.extract_final_symbolic_answer_new(solution_str) #, not_equation = True)

                latex_compliant_substitution_dict = basic_utils.convert_sym_mapping_to_latex(substitution_dict)
                final_substitution_dict = {k: (v if "theta" not in str(k) else v / 180 * np.pi) for k, v in latex_compliant_substitution_dict.items()}
                # Substitute numeric values into the symbolic expression
                generated_answer = "\\boxed{" + str(parsed_exp.subs(final_substitution_dict).evalf()) + "}"

                parse_error = 0
                parsed_sol = str(parsed_exp)
            except Exception as e:
                # print("An error occurred:", e)
                full_error = traceback.format_exc()  # Capture the full stack trace as a string
                parse_error = 1
                parsed_sol = full_error  # Assign the full error message to parsed_sol
                generated_answer = None

        else: # Numeric / Reverse problems
            if has_boxed:
                generated_answer = "\\boxed{" + boxed_list[-1] + "}"
            else:
                generated_answer = solution_str

        if generated_answer is not None:
            try:
                correct, parsed_sol = basic_utils.math_verify_numerical_answer_simple_relative(generated_answer, ground_truth, use_relative_precision=True, max_relative_error=5e-2, mode='relative', relative_to_absolute_threshold=1e-2, data_source=data_source, config = config)
                parsed_sol = parsed_sol[-1]
                parse_error = 0
            except Exception as e:
                # print("An error occurred:", e)
                full_error = traceback.format_exc()  # Capture the full stack trace as a string
                parse_error = 1
                parsed_sol = full_error  # Assign the full error message to parsed_sol
                correct = False
        else:
            correct = False
            
    if correct:
        retval = 1.
    else:
        retval = 0.
    
    results = {
        "score": retval,
        "has_boxed": has_boxed,
        "parse_error": parse_error,
        "parsed_answer": str(parsed_sol),   # this has to be a string. As the parsed numerical answer sometimes is a float which may cause runtime errors on validation.
        "acc": correct,
        "pred": str(parsed_sol),
    }
    return results