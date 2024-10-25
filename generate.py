import gc
import json
import os
import re
import sys

import fire
import gradio as gr
import torch
import transformers
import yaml
from peft import PeftModel
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tqdm import tqdm
from transformers import (
    CodeLlamaTokenizer,
    GenerationConfig,
    LlamaForCausalLM,
    LlamaTokenizer,
    AutoTokenizer, 
    AutoModelForCausalLM,
)

from src.utils.config_loader import Config
from src.utils.callbacks import Iteratorize, Stream
from src.utils.prompter import Prompter
from src.utils.response_parser import (
    get_predictions_labels_search,
    get_predictions_labels_split,
)
from src.utils.plot import scatter_hist

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

try:
    if torch.backends.mps.is_available():
        device = "mps"
except:  
    pass


def main(
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

    base_model = cfg.base_model or os.environ.get("BASE_MODEL", "")
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='huggyllama/llama-7b'"

    prompter = Prompter(cfg.prompt_template_name)
    if re.search("CodeLlama", cfg.base_model):
        tokenizer = CodeLlamaTokenizer.from_pretrained(cfg.base_model)
    elif re.search("Llama-3", cfg.base_model):
        tokenizer = AutoTokenizer.from_pretrained(cfg.base_model)
    else:
        tokenizer = LlamaTokenizer.from_pretrained(cfg.base_model)
    summary_file = os.path.join(cfg.output_dir, "summary.csv")
    error_file = os.path.join(cfg.output_dir, "errors.txt")
    
    if cfg.generate_for_all_checkpoints == 0:
        lora_weights_list = [cfg.lora_weights]
        output_dir_list = [cfg.output_dir]
    else:
        lora_weights_list = [
            os.path.join(cfg.lora_weights, x)
            for x in os.listdir(cfg.lora_weights)
            if os.path.isdir(os.path.join(cfg.lora_weights, x))
            and "checkpoint" in x
        ]
        lora_weights_list = sorted(
            lora_weights_list,
            key=lambda x: int(re.search("checkpoint-(\d+)", x).group(1)),
        )
        output_dir_list = lora_weights_list

    for lora_weights, output_dir in zip(lora_weights_list, output_dir_list):
        try:
            del model
            gc.collect()
            torch.cuda.empty_cache()
        except:
            pass
        if device == "cuda":
            if re.search("Llama-3", base_model):
                model = AutoModelForCausalLM.from_pretrained(
                    base_model,
                    load_in_8bit=cfg.load_8bit,
                    torch_dtype=torch.float16,
                    device_map="auto",
                )
            else:
                model = LlamaForCausalLM.from_pretrained(
                    base_model,
                    load_in_8bit=cfg.load_8bit,
                    torch_dtype=torch.float16,
                    device_map="auto",
                )

            model = PeftModel.from_pretrained(
                model,
                lora_weights,
                torch_dtype=torch.float16,
            )
        elif device == "mps":
            model = AutoModelForCausalLM.from_pretrained(
                base_model,
                device_map={"": device},
                torch_dtype=torch.float16,
            )
            model = PeftModel.from_pretrained(
                model,
                lora_weights,
                device_map={"": device},
                torch_dtype=torch.float16,
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                base_model, device_map={"": device}, low_cpu_mem_usage=True
            )
            model = PeftModel.from_pretrained(
                model,
                lora_weights,
                device_map={"": device},
            )

        model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
        if re.search("CodeLlama", base_model) or re.search("Llama-2", base_model):
            model.config.bos_token_id = 1
            model.config.eos_token_id = 2

        if not cfg.load_8bit:
            model.half() 
        model = model.bfloat16()
        model.eval()
        if torch.__version__ >= "2" and sys.platform != "win32":
            model = torch.compile(model)

        def evaluate(
            instruction,
            input=None,
            temperature=cfg.temperature,
            top_p=0.75,
            top_k=40,
            num_beams=4,
            max_new_tokens=1024,
            stream_output=False,
            **kwargs,
        ):
            prompt = prompter.generate_prompt(instruction, input)
            inputs = tokenizer(prompt, return_tensors="pt")
            input_ids = inputs["input_ids"].to(device)

            if cfg.do_sample:
                generation_config = GenerationConfig(
                    do_sample=cfg.do_sample,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    num_beams=num_beams,
                    pad_token_id = tokenizer.pad_token_id,
                    **kwargs,
                )
            else:
                generation_config = GenerationConfig(
                    do_sample=cfg.do_sample,
                    pad_token_id = tokenizer.pad_token_id,
                    **kwargs,
                )

            generate_params = {
                "input_ids": input_ids,
                "generation_config": generation_config,
                "return_dict_in_generate": True,
                "output_scores": True,
                "max_new_tokens": max_new_tokens,
            }

            if stream_output:
                def generate_with_callback(callback=None, **kwargs):
                    kwargs.setdefault(
                        "stopping_criteria",
                        transformers.StoppingCriteriaList(),
                    )
                    kwargs["stopping_criteria"].append(
                        Stream(callback_func=callback)
                    )
                    with torch.no_grad():
                        model.generate(**kwargs)

                def generate_with_streaming(**kwargs):
                    return Iteratorize(
                        generate_with_callback, kwargs, callback=None
                    )

                with generate_with_streaming(**generate_params) as generator:
                    for output in generator:
                        decoded_output = tokenizer.decode(output)

                        if output[-1] in [tokenizer.eos_token_id]:
                            break

                        yield prompter.get_response(decoded_output)
                return  

            with torch.no_grad():
                generation_output = model.generate(
                    input_ids=input_ids,
                    generation_config=generation_config,
                    return_dict_in_generate=True,
                    output_scores=True,
                    max_new_tokens=max_new_tokens,
                )
            s = generation_output.sequences[0]
            output = tokenizer.decode(s)
            yield prompter.get_response(output)

        if cfg.use_valid_set:
            with open(cfg.valid_data_path, "r") as f:
                valid_dataset = json.load(f)
            if not cfg.do_sample:
                file_name = os.path.join(
                    output_dir, "responses_validation.txt"
                )
            else:
                reg_pattern = re.compile(
                    "responses_validation_temp(\d+\.*\d*)_v(\d+).txt"
                )
                d = (
                    max(
                        [
                            int(reg_pattern.search(x).group(2))
                            for x in os.listdir(output_dir)
                            if reg_pattern.search(x)
                        ]
                    )
                    + 1
                )
                file_name = os.path.join(
                    output_dir,
                    f"responses_validation_temp{cfg.temperature}_v{d}.txt",
                )
            if os.path.exists(file_name):
                continue
            f_out = open(file_name, "w")
            pbar = tqdm(valid_dataset)
            for data in pbar:
                output = " ".join(
                    x
                    for x in evaluate(
                        instruction=data["instruction"],
                        input=data["input"],
                    )
                )
                f_out.write(output)
                f_out.write("\n")
            f_out.close()


        with open(cfg.test_data_path, "r") as f:
            test_dataset = json.load(f)
        

        if not cfg.do_sample:
            file_name = os.path.join(output_dir, "responses.txt")
        else:
            reg_pattern = re.compile("responses_temp(\d+\.*\d*)_v(\d+).txt")
            d = (
                max(
                    [
                        int(reg_pattern.search(x).group(2))
                        for x in os.listdir(output_dir)
                        if reg_pattern.search(x)
                    ]
                )
                + 1
            )
            file_name = os.path.join(
                output_dir, f"responses_temp{cfg.temperature}_v{d}.txt"
            )
        f_out = open(file_name, "w")
        pbar = tqdm(test_dataset)
        for data in pbar:
            output = " ".join(
                x
                for x in evaluate(
                    instruction=data["instruction"],
                    input=data["input"],
                )
            )
            f_out.write(output)
            f_out.write("\n")
        f_out.close()

    f_error_out = open(error_file, "w")
    valid_results = {}
    test_results = {}
    valid_best_r2 = -100000000
    test_best_r2 = 0
    valid_best_r2_checkpoint = ""
    count_not_equal = 0
    for lora_weights, output_dir in zip(lora_weights_list, output_dir_list):
        if cfg.use_valid_set:
            valid_results[lora_weights] = []
            file_name = os.path.join(output_dir, "responses_validation.txt")
            try:
                if cfg.parsing_pattern == "search":
                    predictions, labels = get_predictions_labels_search(
                        file_name,
                        cfg.valid_data_path,
                        reg_patterns=cfg.properties, 
                        numerical=True,
                    )
                elif cfg.parsing_pattern == "split":
                    predictions, labels = get_predictions_labels_split(
                        file_name,
                        cfg.valid_data_path,
                        split_pattern="<MASK_(?:\d+)>",
                        numerical=True,
                    )
                predictions_merged = []
                labels_merged = []
                for i, (pred, label) in enumerate(zip(predictions, labels)):
                    if len(pred) != len(label):
                        count_not_equal += 1
                        f_error_out.write(
                            f"{lora_weights}, label-prediction count mismatch ({i}-th sample, preds:{pred}, labels:{label})\n"
                        )
                        continue
                    predictions_merged += pred
                    labels_merged += label
                valid_results[lora_weights].append(
                    r2_score(labels_merged, predictions_merged)
                )
                valid_results[lora_weights].append(
                    mean_squared_error(labels_merged, predictions_merged) ** 0.5
                )
                valid_results[lora_weights].append(
                    mean_absolute_error(labels_merged, predictions_merged)
                )
                if valid_best_r2 < valid_results[lora_weights][0]:
                    valid_best_r2 = valid_results[lora_weights][0]
                    valid_best_r2_checkpoint = lora_weights
                scatter_file = os.path.join(output_dir, "validation_scatter_plot.png")
                scatter_hist(labels_merged, predictions_merged, "label", "prediction", title = f"R2: {r2_score(labels_merged, predictions_merged)}", file_name = scatter_file)
            except Exception as e:
                f_error_out.write(f"{lora_weights}:\n{e}\n")
                valid_results[lora_weights] += ["ERROR", "ERROR", "ERROR"]
        else:
            valid_results[lora_weights] = [0, 0, 0]
            valid_best_r2_checkpoint = lora_weights

        test_results[lora_weights] = []
        file_name = os.path.join(output_dir, "responses.txt")

        try:
            if cfg.parsing_pattern == "search":
                predictions, labels = get_predictions_labels_search(
                    file_name,
                    cfg.test_data_path,
                    reg_patterns=cfg.properties, 
                    numerical=True,
                )
            elif cfg.parsing_pattern == "split":
                predictions, labels = get_predictions_labels_split(
                    file_name,
                    cfg.test_data_path,
                    split_pattern="<MASK_(?:\d+)>",
                    numerical=True,
                )
        except Exception as e:
            f_error_out.write(f"{lora_weights}:\n{e}\n")
            test_results[lora_weights] += ["ERROR", "ERROR", "ERROR"]
            continue

        predictions_merged = []
        labels_merged = []
        for i, (pred, label) in enumerate(zip(predictions, labels)):
            if len(pred) != len(label):
                count_not_equal += 1
                f_error_out.write(
                    f"{lora_weights}, label-prediction count mismatch ({i}-th sample, preds:{pred}, labels:{label})\n"
                )
                continue
            predictions_merged += pred
            labels_merged += label
        test_results[lora_weights].append(
            r2_score(labels_merged, predictions_merged)
        )
        test_results[lora_weights].append(
            mean_squared_error(labels_merged, predictions_merged) ** 0.5
        )
        test_results[lora_weights].append(
            mean_absolute_error(labels_merged, predictions_merged)
        )
        scatter_file = os.path.join(output_dir, "test_scatter_plot.png")
        scatter_hist(labels_merged, predictions_merged, "label", "prediction", title = f"R2: {r2_score(labels_merged, predictions_merged)}", file_name = scatter_file)

    f_error_out.close()
    f_out = open(summary_file, "w")
    f_out.write(
        "checkpoint,valid R2,valid MSE,valid MAE,test R2,test MSE,test MAE\n"
    )
    for lora_weights in lora_weights_list:
        single_valid_results = ",".join(
            str(x) for x in valid_results[lora_weights]
        )
        single_test_results = ",".join(
            str(x) for x in test_results[lora_weights]
        )
        f_out.write(
            f"{lora_weights},{single_valid_results},{single_test_results}\n"
        )
    f_out.close()
    print(
        "best model & valid r2 & test r2:",
        lora_weights,
        valid_best_r2,
        test_results[valid_best_r2_checkpoint][0],
    )


if __name__ == "__main__":
    fire.Fire(main)
