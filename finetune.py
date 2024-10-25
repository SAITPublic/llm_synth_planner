import datetime
import functools
import json
import os
import re
import sys
from typing import List

import fire
import matplotlib.pyplot as plt
import numpy as np
import torch
import transformers
import yaml
from datasets import load_dataset

num_of_gpus = torch.cuda.device_count()
print("num_of_gpus:", num_of_gpus)
os.system("python -m bitsandbytes")

from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
)
try:
    from peft import prepare_model_for_int8_training
except:
    from peft import prepare_model_for_kbit_training

from transformers import CodeLlamaTokenizer, LlamaForCausalLM, LlamaTokenizer, AutoTokenizer, AutoModelForCausalLM

from src.data.dataset import BaseDataset, DynamicTransformDataset
from src.training.callbacks import UpdateEpochCallback, GPUUsageLogger
from src.utils.config_loader import Config
from src.utils.prompter import Prompter


def train(
    model_config: str,
    dataloader_config: str = None,
):
    with open(model_config, "r") as f:
        config_dict = yaml.safe_load(f)
    if dataloader_config:
        with open(dataloader_config, "r") as f:
            config_dict.update(yaml.safe_load(f))
    cfg = Config(**config_dict)
    cfg.save_file(cfg)

    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        print(cfg)

    assert (
        cfg.base_model
    ), "Please specify a --base_model, e.g. --base_model='huggyllama/llama-7b'"
    gradient_accumulation_steps = cfg.batch_size // cfg.micro_batch_size

    prompter = Prompter(cfg.prompt_template_name)

    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        gradient_accumulation_steps = gradient_accumulation_steps // world_size

    use_wandb = len(cfg.wandb_project) > 0 or (
        "WANDB_PROJECT" in os.environ and len(os.environ["WANDB_PROJECT"]) > 0
    )
    print("use_wandb:", use_wandb)
    if len(cfg.wandb_project) > 0:
        os.environ["WANDB_PROJECT"] = cfg.wandb_project
    if len(cfg.wandb_watch) > 0:
        os.environ["WANDB_WATCH"] = cfg.wandb_watch
    if len(cfg.wandb_log_model) > 0:
        os.environ["WANDB_LOG_MODEL"] = cfg.wandb_log_model
    print(cfg.wandb_project, cfg.wandb_watch, cfg.wandb_log_model)
    print("device_map:", device_map)

    if re.search("Llama-3", cfg.base_model):
        model = AutoModelForCausalLM.from_pretrained(
            cfg.base_model,
            load_in_8bit=True,
            torch_dtype=torch.float16,
            device_map=device_map,
        )
    else:
        model = LlamaForCausalLM.from_pretrained(
            cfg.base_model,
            load_in_8bit=True,
            torch_dtype=torch.float16,
            device_map=device_map,
        )

    if re.search("CodeLlama", cfg.base_model):
        tokenizer = CodeLlamaTokenizer.from_pretrained(cfg.base_model)
    elif re.search("Llama-3", cfg.base_model):
        tokenizer = AutoTokenizer.from_pretrained(cfg.base_model)
    else:
        tokenizer = LlamaTokenizer.from_pretrained(cfg.base_model)
    

    tokenizer.pad_token_id = (
        0  
    )
    tokenizer.padding_side = "left" 

    try:
        model = prepare_model_for_int8_training(model)
    except:
        model = prepare_model_for_kbit_training(model, 8)

    config = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        target_modules=cfg.lora_target_modules,
        lora_dropout=cfg.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)


    if cfg.train_data_path.endswith(
        ".json"
    ) or cfg.train_data_path.endswith(".jsonl"):
        if not cfg.use_valid_set:
            data = load_dataset("json", data_files=cfg.train_data_path)
        else:
            data = load_dataset(
                "json",
                data_files={
                    "train": cfg.train_data_path,
                    "validation": cfg.valid_data_path
                },
            )
    else:
        data = load_dataset(cfg.train_data_path)

    def calculate_statistics(arr):
        mean = np.mean(arr)
        median = np.median(arr)
        std = np.std(arr)
        min_val = np.min(arr)
        max_val = np.max(arr)

        return {
            "Mean": mean,
            "Median": median,
            "Standard Deviation": std,
            "Minimum": min_val,
            "Maximum": max_val,
        }

    plotting = False
    dataset_types = (
        ["train", "validation"] if "validation" in data else ["train"]
    )
    token_length_list = {k: [] for k in dataset_types}
    for dataset_type in dataset_types:
        for data_dict in data[dataset_type]:
            length = 0
            for j in ["output", "instruction", "input"]:
                length += len(tokenizer.tokenize(data_dict[j]))
            token_length_list[dataset_type].append(length)
        print(calculate_statistics(np.array(token_length_list[dataset_type])))
    if plotting:

        def plot_histogram(dic):
            for k, v in dic.items():
                plt.hist(v, bins="auto", label=k)
                plt.title(f"Dataset token length")
                plt.xlabel("token length")
                plt.ylabel("Frequency")
            plt.grid()
            plt.legend()
            plt.show()

        plot_histogram(token_length_list)

    if cfg.resume_from_checkpoint:
        checkpoint_name = os.path.join(
            cfg.resume_from_checkpoint, "pytorch_model.bin"
        )  
        if not os.path.exists(checkpoint_name):
            checkpoint_name = os.path.join(
                cfg.resume_from_checkpoint, "adapter_model.bin"
            ) 
            cfg.resume_from_checkpoint = (
                False  
            )
        if os.path.exists(checkpoint_name):
            print(f"Restarting from {checkpoint_name}")
            adapters_weights = torch.load(checkpoint_name)
            set_peft_model_state_dict(model, adapters_weights)
        else:
            print(f"Checkpoint {checkpoint_name} not found")

    model.print_trainable_parameters()  

    if cfg.val_set_size > 0 and not cfg.use_valid_set:
        train_val = data["train"].train_test_split(
            test_size=cfg.val_set_size, shuffle=True, seed=42
        )
        train_data = (
            BaseDataset(train_val["train"], tokenizer, prompter, cfg)
            if not dataloader_config
            else DynamicTransformDataset(
                train_val["train"], tokenizer, prompter, cfg
            )
        )
        val_data = (
            BaseDataset(train_val["test"], tokenizer, prompter, cfg)
            if not dataloader_config
            else DynamicTransformDataset(
                train_val["test"], tokenizer, prompter, cfg
            )
        )
    elif cfg.use_valid_set:
        train_data = (
            BaseDataset(data["train"], tokenizer, prompter, cfg)
            if not dataloader_config
            else DynamicTransformDataset(
                data["train"], tokenizer, prompter, cfg
            )
        )
        val_data = (
            BaseDataset(data["validation"], tokenizer, prompter, cfg)
            if not dataloader_config
            else DynamicTransformDataset(
                data["validation"], tokenizer, prompter, cfg
            )
        )
        cfg.val_set_size = len(data["validation"])
    else:
        train_data = (
            BaseDataset(data["train"], tokenizer, prompter, cfg)
            if not dataloader_config
            else DynamicTransformDataset(
                data["train"], tokenizer, prompter, cfg
            )
        )
        val_data = None


    if cfg.saved_sample_data:
        selected_samples = [
            dict(sample)
            for sample in data["train"].select(range(cfg.saved_sample_data))
        ]
        if not os.path.exists(cfg.output_dir):
            os.mkdir(cfg.output_dir)
        with open(os.path.join(cfg.output_dir, "sample_data.json"), "w") as f:
            json.dump(selected_samples, f, indent=4)

    if not ddp and torch.cuda.device_count() > 1:
        model.is_parallelizable = True
        model.model_parallel = True

    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=cfg.micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=cfg.warmup_steps,
            num_train_epochs=cfg.num_epochs,
            learning_rate=cfg.learning_rate,
            fp16=True,
            logging_steps=10,
            optim="adamw_torch",
            evaluation_strategy=cfg.evaluation_strategy
            if cfg.val_set_size > 0
            else "no",
            save_strategy=cfg.evaluation_strategy,
            eval_steps=249 if cfg.val_set_size > 0 else None,
            save_steps=249,
            output_dir=cfg.output_dir,
            save_total_limit=cfg.save_total_limit,
            load_best_model_at_end=True if cfg.val_set_size > 0 else False,
            ddp_find_unused_parameters=False if ddp else None,
            group_by_length=cfg.group_by_length,
            report_to="wandb" if use_wandb else "tensorboard",
            run_name=cfg.wandb_run_name if use_wandb else None,
        ),
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
        callbacks=[UpdateEpochCallback, GPUUsageLogger],
    )
    model.config.use_cache = False

    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    with torch.autocast("cuda"):  
        trainer.train(resume_from_checkpoint=cfg.resume_from_checkpoint)

    model.save_pretrained(cfg.output_dir)

    print(
        "\n If there's a warning about missing keys above, please disregard :)"
    )


if __name__ == "__main__":
    fire.Fire(train)
