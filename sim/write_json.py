import os, pathlib, json, random
import hydra
from collections import defaultdict
import glob
from omegaconf import DictConfig
from sim.utils import find_tags
import ipdb
from tqdm import tqdm
st = ipdb.set_trace

def write_json(cfg):
    DSL_directory = cfg.root_dir
    
    dirs = [os.path.join(DSL_directory, _dir) for _dir in os.listdir(DSL_directory) if os.path.isdir(os.path.join(DSL_directory, _dir)) and _dir != "gen_cots"]    # e.g BasicPulleys, IntermediatePulleys, etc.

    sub_dirs = sum([[os.path.join(_dir, _d) for _d in os.listdir(_dir) if os.path.isdir(os.path.join(_dir, _d))] for _dir in dirs], [])   # e.g. BasicPulleys/scene_0, BasicPulleys/scene_1, etc.
    qa_set = defaultdict(list)
    num_questions, valid_files = 0, 0

    for scene_idx, scene in tqdm(enumerate(sub_dirs), total=len(sub_dirs)):
        parent = scene # os.path.join(DSL_directory, scene)
        yaml_files = []
        
        # Search .yaml extensions
        yaml_files.extend(pathlib.Path(parent).rglob("scene_output.yaml"))

        seen = []
        valid_qa = []

        with open(os.path.join(parent, "valid_qs.txt"), "r") as f:
            data = f.read()
            valid_qa = data.split("\n")

        valid_files = len(valid_qa)

        if len(yaml_files) == 0: continue
        for yaml_file in yaml_files:
            ''' +++ 1. collect generated question with numerical answer +++ '''
            # check if question_numerical_answer_pair directory exists in the parent directory
            if cfg.numerical:
                numerical_folders = glob.glob(os.path.join(parent, 'question_numerical_answer_pair*'))
                # loop through all the numerical answer pairs
                for numerical_folder in numerical_folders:
                    numerical_folder_name = numerical_folder.split('/')[-1].replace('question_numerical_answer_pair','')
                    if numerical_folder_name == '':
                        model_name = 'heuristic'
                    else:
                        model_name = numerical_folder_name
                    files = glob.glob(os.path.join(numerical_folder, '*.txt'))  # eg: ["batch_generation_output/IntermediatePulley/scene_3/question_numerical_answer_pair/qa_4_20250325_181815_285338.txt"]
                    for file in files:
                        if file not in valid_qa:
                            continue

                        qa_file = file.split('/')[-1]
                        if qa_file[:4] in seen:
                            continue
                        seen.append(qa_file[:4])

                        try:
                            with open(os.path.join(file), 'r') as f:
                                text = f.read()
                            question = find_tags('problem', text)
                            answer = float(find_tags('answer', text))
                            simulation_mapping = find_tags('simulation_mapping', text)
                            num_questions += 1
                            if scene_idx % 500 == 0:
                                for key, value in qa_set.items():
                                    print(f"model: {key}, num_problems: {len(value)}", end="\t")
                                print()
                            qa_set[model_name].append(
                                {
                                    "text": question,   # problem description
                                    "image": None,
                                    "answer": answer,
                                    "is_symbolic": False,
                                    "simulation_mapping": simulation_mapping,
                                    "model_name": model_name,
                                    "given_variable_mapping": None,    # given variables in the problem
                                    "reference": '/'.join(yaml_file.parts[1:-1]) + '/' + yaml_file.parts[-1].split(".")[0],
                                    "simDSL": str(yaml_file)
                                }
                            )
                            qa_set['all'].append(
                                {
                                    "text": question,   # problem description
                                    "image": None,
                                    "answer": answer,
                                    "is_symbolic": False,
                                    "simulation_mapping": simulation_mapping,
                                    "model_name": model_name,
                                    "given_variable_mapping": None,    # given variables in the problem
                                    "reference": '/'.join(yaml_file.parts[1:-1]) + '/' + yaml_file.parts[-1].split(".")[0],
                                    "simDSL": str(yaml_file)
                                }
                            )                            
                        except:
                            pass
                ''' +++ 2. collect generated question with symbolic answer +++ '''
            
            # check if symbolic_question_answer_pair directory exists in the parent directory
            if cfg.symbolic:
                symbolic_folders = glob.glob(os.path.join(parent, 'symbolic_question_answer_pair*'))
                for symbolic_folder in symbolic_folders:
                    symbolic_folder_name = symbolic_folder.split('/')[-1].replace('symbolic_question_answer_pair','')
                    if symbolic_folder_name == '':
                        model_name = 'heuristic'
                    else:
                        model_name = symbolic_folder_name
                    files = glob.glob(os.path.join(symbolic_folder, '*.txt'))
                    for file in files:
                        if file not in valid_qa:
                            continue
                        try:
                            with open(os.path.join(file), 'r') as f:
                                text = f.read()
                            
                            question = find_tags('problem', text)
                            answer = float(find_tags('answer', text))
                            simulation_mapping = find_tags('simulation_mapping', text)
                            mapping = find_tags('mapping', text)
                            qa_set[model_name].append(
                                {
                                    "text": question,   # problem description
                                    "image": None,
                                    "answer": answer,
                                    "is_symbolic": True,
                                    "simulation_mapping": simulation_mapping,
                                    "model_name": model_name,
                                    "given_variable_mapping": mapping,
                                    "reference": '/'.join(yaml_file.parts[1:-1]) + '/' + yaml_file.parts[-1].split(".")[0],
                                    "simDSL": str(yaml_file)
                                }
                            )
                            
                            qa_set['all'].append({
                                    "text": question,   # problem description
                                    "image": None,
                                    "answer": answer,
                                    "is_symbolic": True,
                                    "simulation_mapping": simulation_mapping,
                                    "model_name": model_name,
                                    "given_variable_mapping": mapping,
                                    "reference": '/'.join(yaml_file.parts[1:-1]) + '/' + yaml_file.parts[-1].split(".")[0],
                                    "simDSL": str(yaml_file)
                                }
                            )                                
                        except:
                            print("not found")
                            pass

    print("num valid files: ", valid_files)
    
    # sort qa_set by model_name
    for key, value in qa_set.items():
        # Split into train (90%) and test (10%) sets
        random.shuffle(value)
        num_problems = len(value)
        num_train = int(0.9 * num_problems)
        
        train_data = value[:num_train]
        test_data = value[num_train:]
        
        if cfg.symbolic:
            prefix = 'symbolic'
        else:
            prefix = 'numerical'
        
        # Save train set
        with open(os.path.join(DSL_directory, f'{prefix}_problems_{key}_train_without_shortcut.json'), 'w') as f:
            print(f"model: {key}, num_train: {len(train_data)}")
            json.dump(train_data, f, indent=4)
            
        # Save test set    
        with open(os.path.join(DSL_directory, f'{prefix}_problems_{key}_test_without_shortcut.json'), 'w') as f:
            print(f"model: {key}, num_test: {len(test_data)}")
            json.dump(test_data, f, indent=4)
        
        # save a small test set
        with open(os.path.join(DSL_directory, f'{prefix}_problems_{key}_test_small_without_shortcut.json'), 'w') as f:
            print(f"model: {key}, num_test: {len(test_data)}")
            json.dump(test_data[:250], f, indent=4)
        
    print(f"stored in {DSL_directory}")

@hydra.main(config_path="../config", config_name="config")
def main(cfg: DictConfig):
    write_json(cfg)
if __name__ == "__main__":
    main()