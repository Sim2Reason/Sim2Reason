import ipdb
import json
import re
import numpy as np
import traceback
import polars as pl
import os
import sys
from importlib.metadata import version, PackageNotFoundError
import multiprocessing
import time
import signal
import functools
from llm.utils import basic_utils
st = ipdb.set_trace

class FallbackConfig:
    debug_print=False
    conservative_reward=False


def timeout_handler(signum, frame):
    raise TimeoutError("Operation timed out")

def timeout_decorator(seconds):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Set the signal handler and a 5-second alarm
            old_handler = signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(seconds)
            try:
                result = func(*args, **kwargs)
            finally:
                signal.alarm(0)  # Disable the alarm
                signal.signal(signal.SIGALRM, old_handler)  # Restore old handler
            return result
        return wrapper
    return decorator


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




def _compute_score_worker(solution_str, ground_truth, data_source, extra_info, config, result_queue):
    try:
        result = compute_score_function(solution_str, ground_truth, data_source, extra_info, config)
        result_queue.put(('success', result))
    except Exception as e:
        result_queue.put(('error', e))

def compute_score_with_timeout(solution_str, ground_truth, data_source, extra_info, config=None, timeout_seconds=10):
    """
    Compute the score for a given solution, with a timeout.
    If the computation exceeds timeout_seconds, returns a timeout result.
    """
    if config.debug_print:
        print(f"DEBUG: compute_score called with timeout_seconds={timeout_seconds}")
    result_queue = multiprocessing.Queue()
    process = multiprocessing.Process(
        target=_compute_score_worker,
        args=(solution_str, ground_truth, data_source, extra_info, config, result_queue)
    )
    process.start()
    process.join(timeout_seconds)
    if process.is_alive():
        print(f"Timeout: compute_score_function exceeded {timeout_seconds} seconds.")
        process.terminate()
        process.join(1)
        if process.is_alive():
            process.kill()
        return {
            "score": 0,
            "has_boxed": False,
            "parse_error": 1,
            "parsed_answer": "TimeoutError: compute_score_function exceeded timeout.",
            "timed_out": True,
        }
    try:
        status, result = result_queue.get_nowait()
        if status == 'success':
            # Add 'timed_out': False to the result if not already present
            if isinstance(result, dict):
                result = dict(result)  # copy to avoid mutating original
                result.setdefault("timed_out", False)
            return result
        else:
            print(f"Error in compute_score_function: {result}")
            return {
                "score": 0,
                "has_boxed": False,
                "parse_error": 1,
                "parsed_answer": str(result),
            }
    except Exception as e:
        print(f"Error in compute_score_function (result retrieval): {e}")
        return {
            "score": 0,
            "has_boxed": False,
            "parse_error": 1,
            "parsed_answer": str(e),
        }

def compute_score(solution_str, ground_truth, data_source, extra_info, config=None):
    return compute_score_function(solution_str, ground_truth, data_source, extra_info, config)


def compute_score_function(solution_str, ground_truth, data_source, extra_info, config=None):
    """
    Verifies the ground truth against the found solution for problems that
    have an equation as the answer.
    """
    # st()
    # assert config is not None, "config must be provided"
    if config is None:
        config = FallbackConfig
    
    if config.debug_print:
        print("DEBUG: extracting bracket content")
    try:
        boxed_list = basic_utils.extract_bracket_content(solution_str)
        if config.debug_print:
            print("DEBUG: extracting bracket content done")
    except Exception as e:
        if config.debug_print:
            print(f"DEBUG: Error extracting bracket content: {e}")
        boxed_list = []
        return {
            "score": 0,
            "has_boxed": False,
            "parse_error": 1,
            "parsed_answer": f"Error extracting bracket content: {str(e)}",
            "timed_out": True,
            "acc": 0,
            "pred": str(solution_str),
        }

    has_boxed = len(boxed_list) > 0
    if config.debug_print:
        print(f"DEBUG: has_boxed = {has_boxed}, boxed_list length = {len(boxed_list)}")
        print(f"DEBUG: data_source = {data_source}")
    
    if "validation" in data_source or "hcv" in data_source or "ipho" in data_source or "Types_1-5_Full_Dataset" in data_source:
        try:
            if config.debug_print:
                print("DEBUG: About to call math_verify_numerical_answer_simple_relative", solution_str, ground_truth, data_source)
            correct, parsed_sol = basic_utils.math_verify_numerical_answer_simple_relative(
                solution_str if not has_boxed else f"\\boxed{{{boxed_list[-1]}}}", ground_truth, 
                use_relative_precision=False, max_relative_error=0.01, 
                mode='relative', relative_to_absolute_threshold=1e-2, 
                data_source=data_source, config=config
            )
            if config.debug_print:
                print("DEBUG: math_verify_numerical_answer_simple_relative done")
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
        if config.debug_print:
            print("DEBUG: Entering synthetic numerical or symbolic branch")
        if "symbolic" in data_source:
            if config.debug_print:
                print("DEBUG: Processing symbolic data source")
            # For symbolic problems, we need to parse the solution string
            try:
                if config.debug_print:
                    print("DEBUG: Loading given_variable_mapping")
                given_variable_mapping_str = extra_info['given_variable_mapping']
                given_variable_mapping = json.loads(given_variable_mapping_str)
                given_variable_mapping = {k:str(v).split(" ")[0] for k,v in given_variable_mapping.items()}
                given_variable_mapping["g"] = "9.81"
                if config.debug_print:
                    print("DEBUG: given_variable_mapping loaded successfully")

                # Convert given variables to float
                substitution_dict = {}
                for k, v in given_variable_mapping.items():
                    substitution_dict[k] = float(v)
                if config.debug_print:
                    print("DEBUG: substitution_dict created")

                if config.debug_print:
                    print("DEBUG: About to call extract_final_symbolic_answer_new")
                parsed_exp = basic_utils.extract_final_symbolic_answer_new(
                    solution_str
                )
                if config.debug_print:
                    print("DEBUG: extract_final_symbolic_answer_new completed")

                if config.debug_print:
                    print("DEBUG: About to call convert_sym_mapping_to_latex")
                latex_compliant_substitution_dict = basic_utils.convert_sym_mapping_to_latex(
                    substitution_dict
                )
                if config.debug_print:
                    print("DEBUG: convert_sym_mapping_to_latex completed")
                
                final_substitution_dict = {k: (v if "theta" not in str(k) else v / 180 * np.pi) for k, v in latex_compliant_substitution_dict.items()}
                if config.debug_print:
                    print("DEBUG: final_substitution_dict created")
                
                def safe_sympy_ops(expr, subs_dict):
                    return str(expr.subs(subs_dict).evalf())
                
                if config.debug_print:
                    print("DEBUG: About to perform sympy substitution and evaluation")
                generated_answer = "\\boxed{" + safe_sympy_ops(parsed_exp, final_substitution_dict) + "}"
                if config.debug_print:
                    print("DEBUG: sympy operations completed")

                parse_error = 0
                parsed_sol = str(parsed_exp)
            except Exception as e:
                if config.debug_print:
                    print(f"DEBUG: Error in symbolic processing: {e}")
                full_error = traceback.format_exc()  # Capture the full stack trace as a string
                parse_error = 1
                parsed_sol = full_error  # Assign the full error message to parsed_sol
                generated_answer = None

        else: # Numeric / Reverse problems
            if config.debug_print:
                print("DEBUG: Processing numeric/reverse data source")
            if has_boxed:
                generated_answer = "\\boxed{" + boxed_list[-1] + "}"
                if config.debug_print:
                    print(f"DEBUG: Using boxed answer: {generated_answer}")
            else:
                generated_answer = solution_str
                if config.debug_print:
                    print(f"DEBUG: Using solution string as answer: {generated_answer}")

        if generated_answer is not None:
            if config.debug_print:
                print("DEBUG: About to verify generated answer")
            try:
                correct, parsed_sol = basic_utils.math_verify_numerical_answer_simple_relative(
                    generated_answer, ground_truth, 
                    use_relative_precision=True, max_relative_error=5e-2, 
                    mode='relative', relative_to_absolute_threshold=1e-2, 
                    data_source=data_source, config=config
                )
                parsed_sol = parsed_sol[-1]
                parse_error = 0
                if config.debug_print:
                    print("DEBUG: Answer verification completed successfully")
            except Exception as e:
                if config.debug_print:
                    print(f"DEBUG: Error in answer verification: {e}")
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
    
    if config.debug_print:
        print(f"DEBUG: Final result - score: {retval}, correct: {correct}, parse_error: {parse_error}")
    
    results = {
        "score": retval,
        "has_boxed": has_boxed,
        "parse_error": parse_error,
        "parsed_answer": str(parsed_sol),   # this has to be a string. As the parsed numerical answer sometimes is a float which may cause runtime errors on validation.
        "acc": correct,
        "pred": str(parsed_sol),
    }
    if config.debug_print:
        print("DEBUG: Returning results")
    return results
