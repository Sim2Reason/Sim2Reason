<div align="center">

<!-- TITLE -->
# **Solving Physics Olympiad via Reinforcement Learning on Physics Simulators**

**[Mihir Prabhudesai](https://mihirp1998.github.io/)<sup>\*,†</sup>, [Aryan Satpathy](https://aryan-satpathy.github.io/)<sup>\*,†</sup>, [Yangmin Li](https://www.linkedin.com/in/yamy12344/)<sup>†</sup>, [Zheyang Qin](https://qinowen.github.io/)<sup>†</sup>,<br> [Nikash Bhardwaj](https://nikashbhardwaj.com/), [Amir Zadeh](https://sim2reason.github.io)<sup>λ</sup>, [Chuan Li](https://sim2reason.github.io)<sup>λ</sup>, [Katerina Fragkiadaki](https://www.cs.cmu.edu/~katef/), [Deepak Pathak](https://www.cs.cmu.edu/~dpathak/)**

Carnegie Mellon University &nbsp;·&nbsp; <sup>λ</sup>Lambda

<sup>\*</sup>Project co-leads & Equal contribution &nbsp;·&nbsp; <sup>†</sup>Core contributors

<br>

<!-- BADGES -->
[![Website](https://img.shields.io/badge/🌎-Website-blue.svg)](http://sim2reason.github.io)
[![Arxiv](https://img.shields.io/badge/arXiv-Paper-b31b1b.svg)](https://arxiv.org/abs/2604.11805)

<!-- Teaser -->
<video src="https://github.com/user-attachments/assets/250d8b73-9728-46e6-aa4c-3a962a74fbf1
" autoplay loop muted playsinline width="100%"></video>

</div>

*We present **SIM2REASON**: a method for turning physics simulators into scalable generators of question–answer pairs to improve LLM reasoning, removing the need of human annotation in the data-generation pipeline. The core idea is to **structure the randomization with a domain-specific language (DSL)** and use it to procedurally generate reasoning problems, as illustrated in the examples above. LLMs finetuned on this synthetic data get **zero-shot improvement on real world benchmarks** such as International Physics Olympiad.*
## Abstract

We have witnessed remarkable advances in LLM reasoning capabilities with the advent of DeepSeek-R1. However, much of this progress has been fueled by the abundance of internet question–answer (QA) pairs—a major bottleneck going forward, since such data is limited in scale and concentrated mainly in domains like mathematics. In contrast, other sciences such as physics lack large-scale QA datasets to effectively train reasoning-capable models. In this work, we show that physics simulators can serve as a powerful alternative source of supervision for training LLMs for physical reasoning. We generate random scenes in physics engines, create synthetic question–answer pairs from simulated interactions, and train LLMs using reinforcement learning on this synthetic data. Our models exhibit zero-shot sim-to-real transfer to real-world physics benchmarks: for example, training solely on synthetic simulated data improves performance on IPhO (International Physics Olympiad) problems by 5-10 percentage points across model sizes. These results demonstrate that physics simulators can act as scalable data generators, enabling LLMs to acquire deep physical reasoning skills beyond the limitations of internet-scale QA data. Code available at: `https://sim2reason.github.io/`.

## Installation
Create an environment for data generation with the following commands:
```
conda create -n pho_data python=3.11
conda activate pho_data

pip install bpy
pip install mujoco ImageIO
pip install ipdb scipy tabulate pandas matplotlib
pip install hydra-core omegaconf
pip install tqdm wandb

pip install sympy math_verify

pip install transformers pyarrow
```

Create an environment for training with the following commands:
```
conda create -n pho_training python=3.12
conda activate pho_training

pip install vllm==0.8.5.post1
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp312-cp312-linux_x86_64.whl
pip install math-verify==0.7.0 
pip install latex2sympy2_extended==1.0.9
pip install antlr4-python3-runtime==4.11.0
pip install polars ipdb
cd verl_v4
pip install -e .
pip install transformers==4.52.3
pip install setuptools==69.5.1
```

## Preparing Data
Set path to store data and checkpoints:
```bash
export PHO_DATA=<path to folder>
export PHO_CHECKPOINT_DIR=<path to folder>
export DATA_VERSION=v1
```

Generate synthetic scenes by running the following command:
```bash
python -m sim.scene_generator scene_generation.num_scenes=1000 scene_generation=all data_version=$DATA_VERSION
```

Generate QA pairs and filter shortcut questions by running the following command:
```bash
python -m sim.qa_gen_rule data_version=$DATA_VERSION gpt_nlq.num_generations_per_problem=30 numerical=True symbolic=True reverse=False
python -m sim.create_child_scenes data_version=$DATA_VERSION
python -m sim.qa_gen_rule data_version=$DATA_VERSION numerical=True symbolic=True reverse=False gpt_nlq.build_child_scenes=True
```

Preprocess generated QA pairs into format suitable for verl by running the following command:
```bash
python -m sim.write_json data_version=$DATA_VERSION numerical=True symbolic=False reverse=False
python -m sim.write_json data_version=$DATA_VERSION numerical=False symbolic=True reverse=False

python -m llm.preprocess_json_to_parquet --json_names numerical_problems_all_train_without_shortcut.json numerical_problems_all_test_without_shortcut.json symbolic_problems_all_train_without_shortcut.json symbolic_problems_all_test_without_shortcut.json  --data_version "$DATA_VERSION" --no_extra_instruction
```

Upon successfully running the above, `<PHO_DATA>/<DATA_VERSION>` should be populated with two files: `train_<DATA_VERSION>_rl.parquet` and `test_<DATA_VERSION>_rl.parquet`.

## Training
```bash
python -m verl_v4.recipe.dapo.main_dapo exps="[dapo_32b,syn_data,q2.5_14b,gspo,use_kl]"  trainer.n_gpus_per_node=8 trainer.val_before_train=False  sp_size=8 gen_tp=8 trainer.total_epochs=10
```

This command finetunes Qwen2.5 14B Instruct on the generated synthetic data using DAPO algorithm. Training info is logged in wandb by default in the project named `verl`, and checkpoints are saved at `PHO_CHECKPOINT_DIR/<run_name>`. By default, only latest global step checkpoint is saved, deleting the old checkpoints to save storage. 

**Pretrained Checkpints:** We provide our pretrained checkpoints for various Qwen models in [HuggingFace](https://huggingface.co/collections/asatpath/sim2reason).

## Evaluation
Evaluation requires first converting the checkpoint to HuggingFace format by running the following command:
```bash
export ACTOR_DIR=$PHO_CHECKPOINT_DIR/<run_name>/global_step_<step>/actor
python verl_v4/scripts/model_merger.py merge \
    --backend fsdp \
    --local_dir "$ACTOR_DIR" \
    --target_dir "$TARGET_DIR"
```

To evaluate the model's performance on IPhO, download the val set from [HF link](https://huggingface.co/collections/asatpath/sim2reason) to `PHO_DATA/ipho/ipho_numeric_validation_no_instruct.parquet` and run:
```bash
python -m verl_v4.recipe.dapo.simple_eval \
 exps='[dapo_32b,log_all_reward,pass_at_n,simple_eval]' \
   model=qwen2.5-3b-instruct data.val_batch_size=null actor_rollout_ref.rollout.val_kwargs.n=8 \
  max_response_length=30000 trainer.n_gpus_per_node=8 model_name=${MODEL_PATH} \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.85 max_model_len=30000 \
  data.val_files="[${PHO_DATA}/ipho/ipho_numeric_validation_no_instruct.parquet]"
```

To evaluate the model's performance on JEEBench, first download JEEBench by running `setup_jeebench.sh`, then run:
```bash
python -m verl_v4.recipe.dapo.simple_eval \
  exps='[dapo_32b,log_all_reward,pass_at_n,use_JEEBench]' \
  data.val_files="[${PHO_DATA}/JEEBench/dataset.json]" \
  model=qwen2.5-3b-instruct data.val_batch_size=1 actor_rollout_ref.rollout.val_kwargs.n=8 \
  max_response_length=32000 trainer.n_gpus_per_node=8 model_name=${MODEL_PATH} \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.65 max_model_len=32000
```

To evaluate the model's performance on OlympiadBench, run:
```bash
python -m verl_v4.recipe.dapo.simple_eval \
  exps='[dapo_32b,log_all_reward,pass_at_n,use_olympiad_bench]' \
  data.val_files="[${PHO_DATA}/olympiad_bench/OlympiadBench_Dataset/data/OE_TO_physics_en_COMP.json,
  ${PHO_DATA}/olympiad_bench/OlympiadBench_Dataset/data/OE_TO_physics_zh_CEE.json]" \
  model=qwen2.5-3b-instruct data.val_batch_size=1 actor_rollout_ref.rollout.val_kwargs.n=1 \
  max_response_length=32000 trainer.n_gpus_per_node=8 model_name=${MODEL_PATH} \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.65 max_model_len=32000
```

To evaluate the model's performance on [PHYSICS](https://github.com/Zhengsh123/PHYSICS), download the val set from [HF link](https://huggingface.co/datasets/desimfj/PHYSICS/tree/main/data) to `PHO_DATA/PHYSICS/test.jsonl` and run:
```bash
python -m verl_v4.recipe.dapo.simple_eval \
  exps='[dapo_32b,ipho_numeric_val,log_all_reward,pass_at_n,use_PHYSICS]' \
  model=qwen2.5-3b-instruct data.val_batch_size=1 actor_rollout_ref.rollout.val_kwargs.n=8 \
  max_response_length=32000 trainer.n_gpus_per_node=8 model_name=${MODEL_PATH} \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.65 max_model_len=32000
```

Since the verifier used in PHYSICS is not opensource, we use Gemini 2.5 Flash as it is a strong verifier. Set the Google API Key in `llm/utils/basic_utils.py` line 138 before running this evaluation.

## Citation
If you find this work useful in your research, please cite:
```bibtex
@article{prabhudesai2026solving,
  title={Solving Physics Olympiad via Reinforcement Learning on Physics Simulators},
  author={Prabhudesai, Mihir and Satpathy, Aryan and Li, Yangmin and Qin, Zheyang and Bhardwaj, Nikash and Zadeh, Amir and Li, Chuan and Fragkiadaki, Katerina and Pathak, Deepak},
  journal={arXiv preprint arXiv:2604.11805},
  year={2026}
}

```
