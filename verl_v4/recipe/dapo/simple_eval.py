# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Note that we don't combine the main with ray_trainer as ray_trainer is used by other main.
"""

import os
import socket
import ipdb
st = ipdb.set_trace
import hydra
import ray
from omegaconf import OmegaConf
import wandb

from importlib.metadata import version, PackageNotFoundError

from verl.trainer.ppo.reward import load_reward_manager

from .dapo_ray_trainer import RayDAPOTrainer
from verl.utils.tracking import Tracking

@hydra.main(config_path="config", config_name="dapo_trainer", version_base=None)
def main(config):
    assert version("math-verify") == "0.7.0"
    assert version("antlr4-python3-runtime") == "4.11.0"
    assert version("latex2sympy2_extended") == "1.0.9"    
    run_ppo(config)


def run_ppo(config) -> None:
    # print(config.trainer.project_name)
    if not ray.is_initialized():
        # this is for local ray cluster
        env_vars =  {"TOKENIZERS_PARALLELISM": "true", "NCCL_DEBUG": "INFO", "VLLM_LOGGING_LEVEL": "WARN"
                    #  "TORCH_NCCL_AVOID_RECORD_STREAMS": "1",
                    # "VLLM_ATTENTION_BACKEND": "XFORMERS",
                    # "GLOO_SOCKET_IFNAME": "eth0",
                    # "NCCL_SOCKET_IFNAME": "eth0"
                    }
        # cv_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        # os.environ["ENSURE_CUDA_VISIBLE_DEVICES"] = cv_devices
        # env_vars["CUDA_VISIBLE_DEVICES"] = cv_devices
        hostname = socket.gethostname()
        if "mgx" in hostname or "gb" in hostname:
            env_vars["GLOO_SOCKET_IFNAME"] = "bond0"            
        
        if "gb" in hostname:
            env_vars["NCCL_DEBUG"] = "INFO"
        
        env_vars["HYDRA_FULL_ERROR"] = "1"
        
        print("env_vars", env_vars)
        print("Initializing Ray")        
        ray.init(
            runtime_env={"env_vars": env_vars},
            num_cpus=config.ray_kwargs.ray_init.num_cpus,
        )

    if OmegaConf.select(config.trainer, "profile_steps") is not None and len(OmegaConf.select(config.trainer, "profile_steps")) > 0:
        nsight_options = OmegaConf.to_container(config.trainer.controller_nsight_options)
        runner = TaskRunner.options(runtime_env={"nsight": nsight_options}).remote()
    else:
        runner = TaskRunner.remote()
    ray.get(runner.run.remote(config))


@ray.remote(num_cpus=1)  # please make sure main_task is not scheduled on head
class TaskRunner:
    def run(self, config):
        # print initial config
        from pprint import pprint

        from omegaconf import OmegaConf
        OmegaConf.register_new_resolver("add", lambda x, y: x + y)
        OmegaConf.register_new_resolver("multiply", lambda x, y: x * y)


        slurm_job_id = os.environ.get('SLURM_JOB_ID', None)
        if slurm_job_id:
            print(f"Running as SLURM job: {slurm_job_id}")
        else:
            print("Not running as a SLURM job")
        
        config.slurm_job_id = slurm_job_id    

        if isinstance(config.data.train_files, str):
            config.actual_train_filename = config.data.train_files.split("/")[-1]
        else:
            config.actual_train_filename = [filename.split("/")[-1] for filename in config.data.train_files]
        
        if isinstance(config.data.val_files, str):
            config.actual_val_filename = config.data.val_files.split("/")[-1]
        else:
            config.actual_val_filename = [filename.split("/")[-1] for filename in config.data.val_files]
        

        from verl.utils.fs import copy_to_local

        print(f"TaskRunner hostname: {socket.gethostname()}, PID: {os.getpid()}")

        pprint(OmegaConf.to_container(config, resolve=True))  # resolve=True will eval symbol values
        OmegaConf.resolve(config)

        # download the checkpoint from hdfs
        local_path = copy_to_local(config.actor_rollout_ref.model.path)

        # instantiate tokenizer
        from verl.utils import hf_processor, hf_tokenizer

        tokenizer = hf_tokenizer(local_path)
        processor = hf_processor(local_path, use_fast=True)  # used for multimodal LLM, could be none

        from verl.single_controller.ray import RayWorkerGroup

        # define worker classes
        if config.actor_rollout_ref.actor.strategy in {"fsdp", "fsdp2"}:
            assert config.critic.strategy in {"fsdp", "fsdp2"}

            from verl.workers.fsdp_workers import ActorRolloutRefWorker, CriticWorker

            ray_worker_group_cls = RayWorkerGroup

        elif config.actor_rollout_ref.actor.strategy == "megatron":
            assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
            from verl.workers.megatron_workers import ActorRolloutRefWorker, CriticWorker

            ray_worker_group_cls = RayWorkerGroup

        else:
            raise NotImplementedError

        from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role

        role_worker_mapping = {
            Role.ActorRollout: ray.remote(ActorRolloutRefWorker),
            Role.Critic: ray.remote(CriticWorker),
        }

        global_pool_id = "global_pool"
        resource_pool_spec = {
            global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
        }
        mapping = {
            Role.ActorRollout: global_pool_id,
            Role.Critic: global_pool_id,
        }

        # we should adopt a multi-source reward function here
        # - for rule-based rm, we directly call a reward score
        # - for model-based rm, we call a model
        # - for code related prompt, we send to a sandbox if there are test cases
        # - finally, we combine all the rewards together
        # - The reward type depends on the tag of the data
        if config.reward_model.enable:
            if config.reward_model.strategy in {"fsdp", "fsdp2"}:
                from verl.workers.fsdp_workers import RewardModelWorker
            elif config.reward_model.strategy == "megatron":
                from verl.workers.megatron_workers import RewardModelWorker
            else:
                raise NotImplementedError
            role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
            mapping[Role.RewardModel] = global_pool_id

        # reference model
        if config.algorithm.use_kl_in_reward or config.actor_rollout_ref.actor.use_kl_loss:
            role_worker_mapping[Role.RefPolicy] = ray.remote(ActorRolloutRefWorker)
            mapping[Role.RefPolicy] = global_pool_id

        reward_fn = load_reward_manager(
            config,
            tokenizer,
            0,
            max_resp_len=config.data.max_response_length,
            overlong_buffer_cfg=config.reward_model.overlong_buffer,
            simple_eval=True,
        )

        # Note that we always use function-based RM for validation
        val_reward_fn = load_reward_manager(
            config,
            tokenizer,
            1,
            max_resp_len=config.data.max_response_length,
            overlong_buffer_cfg=config.reward_model.overlong_buffer,
            simple_eval=True,
        )
        resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)

        if config.solve_locally:
            from vllm import LLM, SamplingParams
            # cv_devices = os.environ.get("ENSURE_CUDA_VISIBLE_DEVICES", "0,1")
            # # cv_devices = "0,1,2,3,4,5,6,7"
            # cv_devices = [int(device) for device in cv_devices.split(",")]
            
            num_gpus = config.trainer.n_gpus_per_node
            num_nodes = config.trainer.nnodes        
            sampling_params = SamplingParams(
                n=config.actor_rollout_ref.rollout.val_kwargs.n,
                max_tokens=config.actor_rollout_ref.rollout.response_length,
            )        
            config.actor_rollout_ref.model.path = config.model_name
            try:
                llm = LLM(model=config.model_name,tensor_parallel_size=num_gpus*num_nodes, max_model_len=config.max_model_len, distributed_executor_backend="ray", gpu_memory_utilization=config.actor_rollout_ref.rollout.gpu_memory_utilization)  # Changes for multi node 
            except Exception as e:
                print(f"Error initializing LLM: {e}")
                # st()
                llm = None
                sampling_params = None
                # llm = LLM(model=config.model_name, tensor_parallel_size=num_gpus, max_model_len=config.max_model_len, dtype=torch.float16)           
        else:
            llm = None
            sampling_params = None    

        if config.get("use_olympiad_bench", False):
            config.trainer.project_name = "olympiad-bench"
            wandb.init(project=config.trainer.project_name, name=config.trainer.experiment_name, config = OmegaConf.to_container(config, resolve=True))
            from verl_v4.recipe.dapo.olympiad_bench import benchmark
            benchmark(config, llm, sampling_params, wandb)  # pass wandb to log inside benchmark
            return
    
        if config.get("use_PHYSICS", False):
            config.trainer.project_name = "PHYSICS"
            wandb.init(project=config.trainer.project_name, name=config.trainer.experiment_name, config = OmegaConf.to_container(config, resolve=True))
            from verl_v4.recipe.dapo import PHYSICS
            PHYSICS.benchmark(config, llm, sampling_params, wandb)  # pass wandb to log inside benchmark
            return
        
        if config.get("use_JEEBench", False):
            config.trainer.project_name = "JEEBench"
            wandb.init(project=config.trainer.project_name, name=config.trainer.experiment_name, config = OmegaConf.to_container(config, resolve=True))
            from verl_v4.recipe.dapo import JEEBench
            JEEBench.benchmark(config, llm, sampling_params, wandb)  # pass wandb to log inside benchmark
            return
        
        trainer = RayDAPOTrainer(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            reward_fn=reward_fn,
            val_reward_fn=val_reward_fn,
            device_name=config.trainer.device,
            simple_eval=True,
        )
        # trainer.init_workers()
        trainer.config.trainer.project_name = "simple-eval"
        
        logger = Tracking(
            project_name=trainer.config.trainer.project_name,
            experiment_name=trainer.config.trainer.experiment_name,
            default_backend=trainer.config.trainer.logger,
            config=OmegaConf.to_container(trainer.config, resolve=True),
        )

        if 'wandb' in logger.logger:
            trainer.config.trainer.default_local_dir = os.path.join(trainer.config.trainer.default_local_dir,logger.logger['wandb'].run.name)
        
        val_metrics = trainer._validate(llm_instance=llm, sampling_params=sampling_params ,logger=logger)
        logger.log(data=val_metrics, step=0)

if __name__ == "__main__":
    main()
