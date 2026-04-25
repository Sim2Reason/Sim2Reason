# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
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

import copy
import logging
import os
import re
import ipdb
st = ipdb.set_trace
from collections import defaultdict
from typing import Optional

import datasets
import numpy as np
import torch
from omegaconf import DictConfig, ListConfig
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, ProcessorMixin

import verl.utils.torch_functional as verl_F
from verl.utils.model import compute_position_id_with_mask

logger = logging.getLogger(__name__)


def collate_fn(data_list: list[dict]) -> dict:
    """
    Collate a batch of sample dicts into batched tensors and arrays.

    Args:
        data_list: List of dicts mapping feature names to torch.Tensor or other values.

    Returns:
        Dict where tensor entries are stacked into a torch.Tensor of shape
        (batch_size, \*dims) and non-tensor entries are converted to
        np.ndarray of dtype object with shape (batch_size,).
    """
    tensors = defaultdict(list)
    non_tensors = defaultdict(list)

    for data in data_list:
        for key, val in data.items():
            if isinstance(val, torch.Tensor):
                tensors[key].append(val)
            else:
                non_tensors[key].append(val)

    for key, val in tensors.items():
        tensors[key] = torch.stack(val, dim=0)

    for key, val in non_tensors.items():
        non_tensors[key] = np.fromiter(val, dtype=object, count=len(val))

    return {**tensors, **non_tensors}


class RLHFDataset(Dataset):
    """
    Load and preprocess RLHF data from Parquet files.

    - Caches files locally.
    - Reads into a HuggingFace Dataset and tokenizes prompts.
    - Optionally handles images/videos via a ProcessorMixin.
    - Filters prompts over a max length.
    - Supports resuming from checkpoints.

    Args:
        data_files (str or list): Path(s) to Parquet file(s).
        tokenizer (PreTrainedTokenizer): For the tokenization of text to token IDs.
        config (DictConfig): Options like cache_dir, prompt_key, max_prompt_length, truncation, etc.
        processor (ProcessorMixin, optional): Multimodal preprocessor for images/videos.
    """

    system_text = """You are a helpful assistant that can solve the given physics question step by step with the help of python interpreter tool.
You can use the python interpreter tool to do arithmetic calculations and solve equations.
To use the python interpreter tool, you need to write a python code snippet inside <python>...</python> tags. It will be exectuted and the result will be returned to you enclosed in <result>...</result> tags."""
    
    def __init__(
        self,
        data_files: str | list[str],
        tokenizer: PreTrainedTokenizer,
        config: DictConfig,
        processor: Optional[ProcessorMixin] = None,
        is_train: bool = True,
        engine: str = "verl",
    ):
        is_eval_dataset = not is_train
        if not isinstance(data_files, list | ListConfig):
            data_files = [data_files]

        self.data_files = copy.deepcopy(data_files)
        self.original_data_files = copy.deepcopy(data_files)  # use for resume
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        self.cache_dir = os.path.expanduser(config.get("cache_dir", "~/.cache/verl/rlhf"))
        self.prompt_key = config.get("prompt_key", "prompt")
        self.image_key = config.get("image_key", "images")
        self.video_key = config.get("video_key", "videos")
        self.max_prompt_length = config.get("max_prompt_length", 1024)
        self.return_raw_chat = config.get("return_raw_chat", False)
        self.return_full_prompt = config.get("return_full_prompt", False)
        self.truncation = config.get("truncation", "error")
        self.filter_overlong_prompts = config.get("filter_overlong_prompts", True)
        self.valid_scene_types = config.get("valid_scene_types", None)  # only used in training dataset
        if self.valid_scene_types is not None:
            if isinstance(self.valid_scene_types, str):
                self.valid_scene_types = [self.valid_scene_types]
            self.valid_scene_types = [scene_type.lower() for scene_type in self.valid_scene_types]
        self.apply_chat_template_kwargs = config.get("apply_chat_template_kwargs", {})

        self.num_workers = config.get("filter_overlong_prompts_workers", max(1, os.cpu_count() // 4))
        self.num_workers = min(self.num_workers, os.cpu_count())
        self.use_shm = config.get("use_shm", False)
        self.chat_template_func = config.get("chat_template_func", None)
        self.need_tools_kwargs = config.get("need_tools_kwargs", False)
        self.filter_prompts = config.get("filter_prompts", True)
        self.serialize_dataset = False
        self.max_samples = config.get("max_examples", None)
        self.valid_data_sources = config.get("valid_data_sources", None)

        self.num_options = config.get("num_options", -1)
        self.add_system_prompt = config.get("add_system_prompt", False)
        self.is_eval_dataset = is_eval_dataset  # Add flag to distinguish eval datasets

        # Print dataset type for num_options feature
        if self.num_options > 0:
            dataset_type = "EVAL" if is_eval_dataset else "TRAIN"
            will_convert = self.num_options > 0 and not self.is_eval_dataset
            print(f"📊 Creating {dataset_type} dataset (num_options={self.num_options}, will_convert={will_convert})")

        self.return_multi_modal_inputs = config.get("return_multi_modal_inputs", True)

        self.engine = engine

        self._download()
        self._read_files_and_tokenize()

    def _download(self, use_origin_parquet=False):
        from verl.utils.fs import copy_to_local

        data_files = self.data_files if not use_origin_parquet else self.original_data_files
        for i, parquet_file in enumerate(data_files):
            self.data_files[i] = copy_to_local(src=parquet_file, cache_dir=self.cache_dir, use_shm=self.use_shm)

    def _read_files_and_tokenize(self):
        dataframes = []
        for parquet_file in self.data_files:
            # read parquet files and cache
            print(f"reading parquet file: {parquet_file}")
            dataframe = datasets.load_dataset("parquet", data_files=parquet_file)["train"]
            # st()
            has_test_split = any(
                doc.get("extra_info", {}).get("split") is not None and doc.get("extra_info", {}).get("split", "").lower() == "test"
                for doc in dataframe.select(range(min(len(dataframe), 100)))  # limit for speed
            )
            # st()
            if self.valid_scene_types is not None and not has_test_split:
                dataframe = dataframe.filter(lambda doc: doc["scene_type"].lower() in self.valid_scene_types)
            if self.valid_data_sources is not None:
                dataframe = dataframe.filter(lambda doc: doc["data_source"].lower() in self.valid_data_sources)
            # st()
            dataframes.append(dataframe)

        try:
            self.dataframe: datasets.Dataset = datasets.concatenate_datasets(dataframes)
        except Exception as e:
            logger.error(f"Error concatenating datasets: {e}")
            def fix_and_merge_dataframes(datasets_list):
                from datasets import Dataset, Features, Sequence, Value
                import pandas as pd
                # Step 1: Convert each Dataset to pandas and clean
                cleaned_pandas = []
                for ds in datasets_list:
                    df = ds.to_pandas()
                    
                    if "scene_type" not in df.columns:
                        df["scene_type"] = "no_scene_type"
                    
                    df["scene_type"] = df["scene_type"].astype(str)
                    df["data_source"] = df["data_source"].astype(str)
                    df["ability"] = df["ability"].astype(str)

                    # fix reward_model
                    df["reward_model"] = df["reward_model"].apply(lambda rm: {
                        "ground_truth": str(rm["ground_truth"]),
                        "style": str(rm["style"]),
                    })

                    # fix extra_info
                    df["extra_info"] = df["extra_info"].apply(lambda ei: {
                        **ei,
                        "split": str(ei.get("split", "")),
                    })

                    # fix prompt
                    df["prompt"] = df["prompt"].apply(lambda lst: [
                        {"content": str(item["content"]), "role": str(item["role"])} for item in lst
                    ])

                    cleaned_pandas.append(df)

                # Step 2: Merge into a single pandas dataframe
                full_df = pd.concat(cleaned_pandas, ignore_index=True)
            
                # Step 3: Deal with extra info and prompt
                EXTRA_INFO_FIELDS = {
                    "answer_type": "",
                    "file_name": "",
                    "json_index": 0,
                    "given_variable_mapping": "",
                    "index": 0,
                    "split": "",
                }

                def normalize_extra_info(ei):
                    def safe_convert(key, default_val):
                        if key not in ei:
                            return default_val
                        value = ei.get(key)
                        if value is None:
                            return default_val
                        if isinstance(default_val, str):
                            return str(value)
                        else:
                            # For integer fields, try to convert safely
                            try:
                                return int(value)
                            except (ValueError, TypeError):
                                # If conversion fails (e.g., UUID string), return default
                                return default_val
                    
                    return {
                        key: safe_convert(key, val)
                        for key, val in EXTRA_INFO_FIELDS.items()
                    }

                def normalize_prompt(prompt):
                    return [
                        {
                            "content": str(item.get("content", "")),
                            "role": str(item.get("role", ""))
                        }
                        for item in prompt
                    ] if isinstance(prompt, list) else []
                
                full_df["extra_info"] = full_df["extra_info"].apply(normalize_extra_info)
                full_df["prompt"] = full_df["prompt"].apply(normalize_prompt)

                # Step 3: Define schema explicitly
                extra_info_schema = {
                    "answer_type": Value("string"),
                    "file_name": Value("string"),
                    "json_index": Value("int64"),
                    "given_variable_mapping": Value("string"),
                    "index": Value("int64"),
                    "split": Value("string"),
                }
                features = Features({
                    "scene_type":  Value("string"),
                    "data_source": Value("string"),
                    "prompt":      [{"content": Value("string"), "role": Value("string")}],
                    "ability":     Value("string"),
                    "reward_model": {
                        "ground_truth": Value("string"),
                        "style":        Value("string"),
                    },
                    "extra_info": extra_info_schema,
                })
                # st()

                # Step 4: Reconstruct clean Dataset
                dataset = Dataset.from_pandas(full_df, preserve_index=False)
                # {
                #     'scene_type': Value(dtype='string', id=None), 
                #     'data_source': Value(dtype='string', id=None), 
                #     'prompt': [
                #             {
                #                 'content': Value(dtype='string', id=None), 
                #                 'role': Value(dtype='string', id=None)
                #             }
                #         ], 
                #     'ability': Value(dtype='string', id=None), 
                #     'reward_model': {
                #         'ground_truth': Value(dtype='string', id=None), 
                #         'style': Value(dtype='string', id=None)
                #         }, 
                #     'extra_info': {
                #         'answer_type': Value(dtype='string', id=None), 
                #         'file_name': Value(dtype='string', id=None), 
                #         'given_variable_mapping': Value(dtype='string', id=None), 
                #         'index': Value(dtype='int64', id=None), 
                #         'json_index': Value(dtype='int64', id=None), 
                #         'split': Value(dtype='string', id=None)
                #         }
                # }
                return dataset
                return Dataset.from_pandas(full_df, features=features, preserve_index=False)

            self.dataframe = fix_and_merge_dataframes(dataframes)

            # def find_large_string_fields(ds):
            #     from datasets import Value, Features, Sequence
            #     from datasets.features import LargeList
            #     def recurse(f, path=""):
            #         if isinstance(f, Value) and f.dtype == "large_string":
            #             print("⚠️ large_string at:", path)
            #         elif isinstance(f, dict):
            #             for k, v in f.items():
            #                 recurse(v, path + "." + k if path else k)
            #         elif hasattr(f, "feature"):
            #             recurse(f.feature, path + "[]")

            #     for k, v in ds.features.items():
            #         recurse(v, k)
            
            # for ds in dataframes:
            #     find_large_string_fields(ds)
            # exit(1)

        if self.max_samples is not None:
            n = self.max_samples
            self.dataframe = self.dataframe.select(range(min(n, len(self.dataframe))))
            print(f"Taking first {n} examples, actual dataset len: {len(self.dataframe)}")

        print(f"dataset len: {len(self.dataframe)}")

        # filter out too long prompts
        if self.filter_overlong_prompts:
            tokenizer = self.tokenizer
            processor = self.processor
            prompt_key = self.prompt_key
            image_key = self.image_key
            video_key = self.video_key

            if processor is not None:
                from verl.utils.dataset.vision_utils import process_image, process_video

                def doc2len(doc) -> int:
                    messages = self._build_messages(doc)
                    raw_prompt = self.processor.apply_chat_template(
                        messages, add_generation_prompt=True, tokenize=False, **self.apply_chat_template_kwargs
                    )
                    images = [process_image(image) for image in doc[image_key]] if image_key in doc else None
                    videos = [process_video(video) for video in doc[video_key]] if video_key in doc else None

                    return len(processor(text=[raw_prompt], images=images, videos=videos)["input_ids"][0])

            elif tokenizer is not None:

                def doc2len(doc) -> int:
                    return len(
                        tokenizer.apply_chat_template(
                            doc[prompt_key], add_generation_prompt=True, **self.apply_chat_template_kwargs
                        )
                    )

                dataframe = dataframe.filter(
                    lambda doc: doc2len(doc) <= self.max_prompt_length,
                    num_proc=self.num_workers,
                    desc=f"Filtering prompts longer than {self.max_prompt_length} tokens",
                )

            print(f"filter dataset len: {len(self.dataframe)}")
        
        if 'data_source' in self.dataframe.column_names:
            data_source_counts = set(self.dataframe['data_source'])
            print(f"Data source counts: {data_source_counts}")
        if 'scene_type' in self.dataframe.column_names:
            scene_type_counts = set(self.dataframe['scene_type'])
            print(f"Scene type counts: {scene_type_counts}")
        # st()
        if self.num_options > 0 and not self.is_eval_dataset:
            print(f"✅ Applying num_options conversion to training data (classes: {self.num_options})")
            assert np.ceil(self.num_options) == self.num_options, "num_options must be a positive integer"
            import math
            def process_row(row):
                if "numeric" not in row["data_source"].lower() and "reverse" not in row["data_source"].lower(): 
                    return {**row, "was_successful": False}
                try:
                    num_classes = self.num_options
                    gt_str = row["reward_model"]["ground_truth"]
                    gt = float(gt_str)

                    if gt <= 0 or math.isnan(gt) or math.isinf(gt):
                        raise ValueError(f"Invalid gt value: {gt}")

                    # Compute scaling factor: base^floor(log_base(gt))
                    log_base = math.log(gt, num_classes)
                    exponent = math.floor(log_base)
                    to_mul = False
                    if exponent <= 0:
                        to_mul = True
                        exponent = -exponent
                    scale = num_classes ** exponent

                    normalized = (gt * scale) if to_mul else (gt / scale)
                    label = int(round(normalized))
                    label = label % num_classes

                    instruct = r"""Please compute the value, {op} it by {val}, round to the nearest integer (between 0 - {num_options}), and output the result inside \boxed{{}}."""
                    
                    # Modify prompt to mention scale
                    modified_prompt = row["prompt"].copy()
                    for msg in reversed(modified_prompt):
                        if msg["role"] == "user":
                            msg["content"] += instruct.format(op=['divide','multiply'][to_mul], val=scale, num_options=num_classes)
                            break

                    row["prompt"] = modified_prompt
                    row["reward_model"]["ground_truth"] = str(label)

                    return {**row, "was_successful": True}

                except Exception as e:
                    # print(f"Error processing row: {e}")
                    return {**row, "was_successful": False}
            
            print("Converting numerical answers to multi-class classification...")
            self.dataframe = self.dataframe.map(
                process_row,
                num_proc=self.num_workers,
                desc="Mapping numerical answers to classification",
            )

            n_success = sum(self.dataframe["was_successful"])
            print(f"✅ Converted {n_success/len(self.dataframe)*100:.1f}% of data to classification format")
        elif self.num_options > 0 and self.is_eval_dataset:
            print(f"⏭️  Skipping num_options conversion for evaluation dataset")
        
        if self.config.get("add_tool_agent", False):
            self.dataframe = self.dataframe.add_column("agent_name", ["tool_agent"] * len(self.dataframe))
        
        if self.config.get("add_instruct_prompt", True):
            def process_row(row):
                if 'math_dapo' in row["data_source"].lower(): return {**row}
                
                if "numeric" not in row["data_source"].lower() and "reverse" not in row["data_source"].lower() and "symbol" not in row["data_source"].lower(): 
                    return {**row}
                try:
                    legacy_instruct = "Let's think step by step and output the final answer within \\boxed{}. The final answer should be expressed using the International System of Units (SI) unless stated otherwise."
                    new_instruct = "\nSolve step by step by first defining variables for each body, then deriving the force body diagram equations for each body and then carefully deriving all constraint equations. Finally solve the system of equations and return final answer in \\boxed{}."
                    instruct = legacy_instruct if self.config.get("use_legacy_prompt", False) else new_instruct
                    
                    # Modify prompt to mention scale
                    modified_prompt = row["prompt"].copy()
                    for msg in reversed(modified_prompt):
                        if msg["role"] == "user":
                            msg["content"] += instruct
                            break

                    row["prompt"] = modified_prompt
                    
                    return {**row}

                except Exception as e:
                    # print(f"Error processing row: {e}")
                    return {**row}
                
            def keep(example):
                return abs(float(example["reward_model"]["ground_truth"])) > 1

            print("Adding instruction prompt...")
            self.dataframe = self.dataframe.map(
                process_row,
                num_proc=self.num_workers,
                desc="Adding instruction prompt",
            )

            # self.dataframe = self.dataframe.filter(keep)

        return self.dataframe

    def resume_dataset_state(self):
        self.serialize_dataset = not hasattr(self, "original_data_files")
        # resume dataframe if not it's serialized in data.pt
        if not self.serialize_dataset:
            self._download(use_origin_parquet=True)  # download and resume from original parquet files
            self._read_files_and_tokenize()
        else:
            print(r"old dataloader ckpt file is used, please train from scratch for better ckpt performance")

    def __len__(self):
        return len(self.dataframe)

    def _build_messages(self, example: dict):
        messages: list = example.pop(self.prompt_key)

        if self.image_key in example or self.video_key in example:
            for message in messages:
                content = message["content"]
                content_list = []
                segments = re.split("(<image>|<video>)", content)
                segments = [item for item in segments if item != ""]
                for segment in segments:
                    if segment == "<image>":
                        content_list.append({"type": "image"})
                    elif segment == "<video>":
                        content_list.append({"type": "video"})
                    else:
                        content_list.append({"type": "text", "text": segment})

                message["content"] = content_list

        if not any([message["role"]=="system" for message in messages]) and self.add_system_prompt:
            messages = [{"role": "system", "content": RLHFDataset.system_text}] + messages

        return messages

    def __getitem__(self, item):
        """
        Note that we also return the raw_input_ids so that it can be combined with other chat template
        """
        row_dict: dict = self.dataframe[item] 
        messages = self._build_messages(row_dict)
        
        if self.engine == "tinker":
            row_dict["messages"] = messages
            return row_dict
        
        model_inputs = {}

        if self.processor is not None:
            from verl.utils.dataset.vision_utils import process_image, process_video

            raw_prompt = self.processor.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False, **self.apply_chat_template_kwargs
            )
            multi_modal_data = {}

            images = None
            if self.image_key in row_dict and row_dict.get(self.image_key, None) is not None:
                images = [process_image(image) for image in row_dict.pop(self.image_key)]

                # due to the image key is "image" instead of "images" in vllm, we need to use "image" here
                # link: https://github.com/vllm-project/vllm/blob/3c545c0c3b98ee642373a308197d750d0e449403/vllm/multimodal/parse.py#L205
                multi_modal_data["image"] = images

            videos = None
            if self.video_key in row_dict and row_dict.get(self.video_key, None) is not None:
                videos = [process_video(video) for video in row_dict.pop(self.video_key)]

                # due to the video key is "video" instead of "videos" in vllm, we need to use "video" here
                # link: https://github.com/vllm-project/vllm/blob/3c545c0c3b98ee642373a308197d750d0e449403/vllm/multimodal/parse.py#L205
                multi_modal_data["video"] = [video.numpy() for video in videos]

            model_inputs = self.processor(text=[raw_prompt], images=images, videos=videos, return_tensors="pt")

            input_ids = model_inputs.pop("input_ids")
            attention_mask = model_inputs.pop("attention_mask")

            if "second_per_grid_ts" in model_inputs:
                model_inputs.pop("second_per_grid_ts")

            # There's a trap here, multi_modal_inputs has to be a dict, not BatchFeature
            row_dict["multi_modal_data"] = multi_modal_data

            # We will do batch.union() in the trainer,
            # so we cannot have "multi_modal_inputs" in row_dict if rollout generates new multi_modal_inputs
            if self.return_multi_modal_inputs:
                row_dict["multi_modal_inputs"] = dict(model_inputs)

                # second_per_grid_ts isn't used for training, just for mrope
                row_dict["multi_modal_inputs"].pop("second_per_grid_ts", None)

        else:
            raw_prompt = self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False, **self.apply_chat_template_kwargs
            )
            model_inputs = self.tokenizer(raw_prompt, return_tensors="pt", add_special_tokens=False)
            input_ids = model_inputs.pop("input_ids")
            attention_mask = model_inputs.pop("attention_mask")

        input_ids, attention_mask = verl_F.postprocess_data(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.truncation,
        )

        if self.processor is not None and "Qwen2VLImageProcessor" in self.processor.image_processor.__class__.__name__:
            from verl.models.transformers.qwen2_vl import get_rope_index

            position_ids = [
                get_rope_index(
                    self.processor,
                    input_ids=input_ids[0],
                    image_grid_thw=model_inputs.get("image_grid_thw"),
                    video_grid_thw=model_inputs.get("video_grid_thw"),
                    second_per_grid_ts=model_inputs.get("second_per_grid_ts"),
                    attention_mask=attention_mask[0],
                )
            ]  # (1, 3, seq_len)

        else:
            position_ids = compute_position_id_with_mask(attention_mask)

        row_dict["input_ids"] = input_ids[0]
        row_dict["attention_mask"] = attention_mask[0]
        row_dict["position_ids"] = position_ids[0]

        raw_prompt_ids = self.tokenizer.encode(raw_prompt, add_special_tokens=False)
        if len(raw_prompt_ids) > self.max_prompt_length:
            if self.truncation == "left":
                raw_prompt_ids = raw_prompt_ids[-self.max_prompt_length :]
            elif self.truncation == "right":
                raw_prompt_ids = raw_prompt_ids[: self.max_prompt_length]
            elif self.truncation == "middle":
                left_half = self.max_prompt_length // 2
                right_half = self.max_prompt_length - left_half
                raw_prompt_ids = raw_prompt_ids[:left_half] + raw_prompt_ids[-right_half:]
            elif self.truncation == "error":
                raise RuntimeError(f"Prompt length {len(raw_prompt_ids)} is longer than {self.max_prompt_length}.")

        row_dict["raw_prompt_ids"] = raw_prompt_ids
        # encode prompts without chat template
        if self.return_raw_chat:
            row_dict["raw_prompt"] = messages

        # get prompts with chat template
        if self.return_full_prompt:
            row_dict["full_prompts"] = raw_prompt  # array of strings

        # add index for each prompt
        index = row_dict.get("extra_info", {}).get("index", 0)
        tools_kwargs = row_dict.get("extra_info", {}).get("tools_kwargs", {})
        interaction_kwargs = row_dict.get("extra_info", {}).get("interaction_kwargs", {})
        need_tools_kwargs = row_dict.get("extra_info", {}).get("need_tools_kwargs", self.need_tools_kwargs)
        if need_tools_kwargs and not tools_kwargs:
            logger.warning("tools_kwargs is empty for index {}, data source: {}", index, row_dict["data_source"])
        row_dict["index"] = index
        row_dict["tools_kwargs"] = tools_kwargs
        row_dict["interaction_kwargs"] = interaction_kwargs
        return row_dict

    def __getstate__(self):
        if not self.serialize_dataset:
            state = self.__dict__.copy()

            if "dataframe" in state:
                del state["dataframe"]
            return state

        return self.__dict__.copy()
