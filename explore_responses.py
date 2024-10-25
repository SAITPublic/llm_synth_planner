import os
import gc
import sys
from tqdm import tqdm
import pandas as pd
import numpy as np
import json
import fire
import gradio as gr
import torch
import transformers
import yaml
import random
import re
from collections import defaultdict

from peft import PeftModel
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer, CodeLlamaTokenizer, AutoTokenizer, AutoModelForCausalLM
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

from src.utils.callbacks import Iteratorize, Stream
from src.utils.prompter import Prompter
from src.utils.response_parser import replace_mask_with_predictions, get_predictions_labels_search, calculate_recipe_similarity
from src.utils.plot import scatter_hist

from src.data.dataset import count_decimal_places
from src.utils.config_loader import Config
from src.data.dataset import BaseDataset, DynamicTransformDataset

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

try:
    if torch.backends.mps.is_available():
        device = "mps"
except:  # noqa: E722
    pass


def clean_path(path):
    return path.rstrip('/')

def mean_excluding_one_min_max(row):
    sorted_row = sorted(row)
    if len(sorted_row) > 2:
        return pd.Series(sorted_row[1:-1]).mean()
    else:
        return row.mean()  

def std_excluding_one_min_max(row):
    sorted_row = sorted(row)
    if len(sorted_row) > 2:
        return pd.Series(sorted_row[1:-1]).std()
    else:
        return row.std()  

def main(
    model_config: str,
    exploration_config: str,
    # text_template_config: str, 
    properties = ["Valid Quantum Yield", "Stability Factor"],
):
    print(properties)
    with open(model_config, "r") as f:
        config_dict = yaml.safe_load(f)
    cfg = Config(**config_dict)
    print(cfg)
    with open(exploration_config, "r") as f:
        config_dict = yaml.safe_load(f)
    exp_cfg = Config(flatten = 0, **config_dict)
    print(exp_cfg) 


    cfg.load_8bit = False
    base_model = cfg.base_model or os.environ.get("BASE_MODEL", "")
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='huggyllama/llama-7b'"

    prompter = Prompter(cfg.prompt_template_name)
    tokenizer = AutoTokenizer.from_pretrained(base_model)

    def get_explorer_folder(lora_weights): 
        explorer_folder = os.path.join(lora_weights, "responses_to_input_changes", cfg.data_subfolder_name, "region")
        if not os.path.exists(explorer_folder): os.makedirs(explorer_folder, exist_ok = True)
        return explorer_folder
    
    def evaluate_with_probs(
        instruction,
        input=None,
        temperature=cfg.temperature,
        top_p=0.75,
        top_k=40,
        num_beams=4,
        max_new_tokens=1024,
        stream_output=False,
        normalized_probability = [],
        **kwargs,
    ):
        prompt = prompter.generate_prompt(instruction, input)
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)
        
        if cfg.do_sample:
            generation_config = GenerationConfig(
                do_sample = cfg.do_sample,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                num_beams=num_beams,
                pad_token_id = tokenizer.pad_token_id,
                **kwargs,
            )
        else:
            generation_config = GenerationConfig(
                do_sample = cfg.do_sample,
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

        # Without streaming
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

        softmax = torch.nn.Softmax(dim=-1)
        log_prob_sum = 0
        num_tokens = len(generation_output.scores)

        for i, logits in enumerate(generation_output.scores):
            generated_token_id = generation_output.sequences[0, input_ids.size(1) + i]
            probs = softmax(logits)
            token_prob = probs[0, generated_token_id] 
            log_prob = torch.log(token_prob)
            log_prob_sum += log_prob

        average_log_prob = log_prob_sum / num_tokens
        normalized_probability.append(torch.exp(average_log_prob).item())
        yield prompter.get_response(output)

    lora_weights_list = cfg.lora_weights.split(',')
    if isinstance(properties, str): 
        properties = [properties]
    print("properties:", properties)

    
    sampling_region = {k: getattr(getattr(exp_cfg.target_props, k), "bound") for k in exp_cfg.target_props.__dict__} # keys: ["QET", "ICP mole ratio for Ga/In"]
    print(sampling_region)

    ###################################################################################
    for lora_weights in lora_weights_list:
        try:
            del model
            gc.collect()
            torch.cuda.empty_cache()
        except:
            pass
        if device == "cuda":
            model = AutoModelForCausalLM.from_pretrained(
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

        # unwind broken decapoda-research config
        model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
        model.config.bos_token_id = 1
        model.config.eos_token_id = 2

        if not cfg.load_8bit:
            model.half()  # seems to fix bugs for some users.
        model = model.bfloat16()
        model.eval()
        if torch.__version__ >= "2" and sys.platform != "win32":
            model = torch.compile(model)

        # testing code for readme

        with open(cfg.test_data_path, 'r') as f:
            test_dataset = json.load(f)

        explorer_folder = get_explorer_folder(lora_weights)

        with open(os.path.join(explorer_folder, exploration_config.split('/')[-1]), "w") as f:
            yaml.dump(exp_cfg, f)

        
        if "multi_regions" in exploration_config:
            samples = {k: [random.uniform(*v[j]) for j in range(len(v)) for _ in range(int(exp_cfg.iterations/len(v)))] for k, v in sampling_region.items()}
        else:
            samples = {k: [random.uniform(*v) for _ in range(exp_cfg.iterations)] for k, v in sampling_region.items()}
        print(samples)
        pd.DataFrame(samples, columns = sampling_region.keys()).to_csv(os.path.join(explorer_folder, 'sampled_Y_labels.csv'), index = False)

        for i in range(exp_cfg.iterations):
            # eval_Y_list = []
            org_Y_list = []
            if not cfg.do_sample:
                file_name = os.path.join(explorer_folder, f'responses_iter{i}.txt')
            else:
                reg_pattern = re.compile("responses_iter{i}_temp(\d+\.*\d*)_v(\d+).txt")
                d = max([int(reg_pattern.search(x).group(2)) for x in os.listdir(explorer_folder) if reg_pattern.search(x)]) + 1
                file_name = os.path.join(explorer_folder, f'responses_iter{i}_temp{cfg.temperature}_v{d}.txt')

            f_out = open(file_name, 'w')
            norm_prob_list = []
            pbar = tqdm(enumerate(test_dataset))
            for idx, data in pbar:
                changed_data = {k: v for k, v in data.items()}
                changed_input = changed_data["input"]

                org_labels = []
                for prop in sampling_region.keys():
                    org_label = re.search(getattr(getattr(exp_cfg.target_props, prop), "reg_pattern"), changed_input).group(1)
                    org_labels.append(org_label)
                    changed_input = re.sub(getattr(getattr(exp_cfg.target_props, prop), "reg_pattern"), lambda match: getattr(getattr(exp_cfg.target_props, prop), "replacement").format(str(round(samples[prop][i], count_decimal_places(org_label)))), changed_input)
                     
                if hasattr(cfg, "drop_props_from_input"):
                    changed_data["input"] = changed_input
                    changed_data = DynamicTransformDataset.transform_input_single(changed_data, cfg.drop_props_from_input, "")
                    changed_input = changed_data["input"]
                
                output = ' '.join(x for x in evaluate_with_probs(instruction = changed_data["instruction"], input = changed_input, normalized_probability = norm_prob_list))
                f_out.write(output)
                f_out.write('\n')
                org_Y_list.append(org_labels)
            f_out.close()

            pd.concat([pd.DataFrame(org_Y_list, columns = ["org_" + x for x in sampling_region.keys()]), \
                        pd.DataFrame([[samples[prop][i] for prop in sampling_region.keys()] for _ in range(len(test_dataset))], columns = ["sampled_" + x for x in sampling_region.keys()]), \
                        pd.DataFrame(norm_prob_list, columns = ["norm_probs"])], axis = 1).to_csv(os.path.join(explorer_folder, f'org_Y_sampled_Y_norm_probs_iter{i}.csv'), index = False)
            

    # # ########################### start recipe similarity calculation

    for lora_weights in lora_weights_list:
        explorer_folder = get_explorer_folder(lora_weights)
        for i in range(exp_cfg.iterations):
        #### TO DO: sampling results
            try:
                response_data = os.path.join(explorer_folder, f'responses_iter{i}.txt')
                sub_inputs = replace_mask_with_predictions(response_data, cfg.test_data_path, split_pattern="<MASK_(?:\d+)>")
                recipe_min_distances = calculate_recipe_similarity(response_data, cfg.test_data_path, split_pattern="<MASK_(?:\d+)>")
                with open(os.path.join(explorer_folder, f"recipe_changes_iter{i}.json"), "w") as f:
                    json.dump(recipe_min_distances, f, indent = 4)
            except:
                continue                

    ######################## end recipe similarity calculation
    
    cfg.load_8bit = False
    base_model = cfg.base_pred_model or os.environ.get("BASE_MODEL", "")
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='huggyllama/llama-7b'"

    prompter = Prompter(cfg.prompt_template_name)
    tokenizer = AutoTokenizer.from_pretrained(base_model)

    for pred_lora_weights in cfg.prediction_models:
        pred_lora_weights = clean_path(pred_lora_weights)
        try:
            del model
            gc.collect()
            torch.cuda.empty_cache()
        except:
            pass
        if device == "cuda":
            model = AutoModelForCausalLM.from_pretrained(
                base_model,
                load_in_8bit=cfg.load_8bit,
                torch_dtype=torch.float16,
                device_map="auto",
            )
            model = PeftModel.from_pretrained(
                model,
                pred_lora_weights,
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
                pred_lora_weights,
                device_map={"": device},
                torch_dtype=torch.float16,
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                base_model, device_map={"": device}, low_cpu_mem_usage=True
            )
            model = PeftModel.from_pretrained(
                model,
                pred_lora_weights,
                device_map={"": device},
            )
         # unwind broken decapoda-research config
        model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
        model.config.bos_token_id = 1
        model.config.eos_token_id = 2

        if not cfg.load_8bit:
            model.half()  # seems to fix bugs for some users.
        model = model.bfloat16()
        model.eval()
        if torch.__version__ >= "2" and sys.platform != "win32":
            model = torch.compile(model)

        for lora_weights in lora_weights_list:
            explorer_folder = get_explorer_folder(lora_weights)
            response_data_files = sorted([x for x in os.listdir(explorer_folder) if "responses_iter" in x])
           
            pred_lora_weights_folder = pred_lora_weights.split('/')[-1] if "checkpoint" not in pred_lora_weights.split('/')[-1] else '_'.join(pred_lora_weights.split('/')[-2:])
            eval_folder = os.path.join(explorer_folder, "eval_by_pred_model", pred_lora_weights_folder)
            if not os.path.exists(eval_folder): os.makedirs(eval_folder, exist_ok = True)
            
            for i in range(exp_cfg.iterations):
                response_data = f'responses_iter{i}.txt'
                file_name = os.path.join(eval_folder, "evaluation_" + response_data)
                response_data = os.path.join(explorer_folder, response_data)

                sub_inputs = replace_mask_with_predictions(response_data, cfg.test_data_path, split_pattern="<MASK_(?:\d+)>")
                processed_data_list = []
                eval_Y_df = pd.read_csv(os.path.join(explorer_folder, f'org_Y_sampled_Y_norm_probs_iter{i}.csv'))
               
                # print("len(dataset):", len(dataset))
                for j in range(len(sub_inputs)):
                    formatted_dict = {}
                    formatted_dict["instruction"] = cfg.instruction
                    formatted_dict["input"] = "# Quantum Dot Synthesis Procedure:" + sub_inputs[j].split("# Quantum Dot Synthesis Procedure:")[-1]
                    # formatted_dict["output"] = text[2]
                    processed_data_list.append(formatted_dict)
        
                pbar = tqdm(processed_data_list)
                f_out = open(file_name, 'w')
                norm_prob_list = []
                for data in pbar:
                    output = ' '.join(x for x in evaluate_with_probs(instruction = data["instruction"], input = data["input"], normalized_probability = norm_prob_list))
                    # print(norm_prob_list)
                    f_out.write(output)
                    f_out.write('\n')
                f_out.close()

                reg_patterns = ["Valid Quantum Yield\: (\d+\.*\d*)\s?\%", "Stability Factor\: (\d+\.*\d*)\s?"]
                for property, reg_pattern in zip(properties, reg_patterns):
                    error_file = os.path.join(eval_folder, property + "_errors.txt")
                    performance_file = os.path.join(eval_folder, property + f"_performance_iter{i}.csv")
                    predict_all_file =os.path.join(eval_folder, property + f"_predict_all_iter{i}.csv")

                    try:
                        eval_predictions, _ = get_predictions_labels_search(file_name, None, reg_patterns = [reg_pattern], numerical = True)
                        sampled_Y_list = eval_Y_df["sampled_" + property].tolist()
                        org_Y_list = eval_Y_df["org_" + property].tolist()
                        assert len(eval_predictions) == len(sampled_Y_list), f"{eval_predictions}, \n{sampled_Y_list}, \n{len(eval_predictions)}, {len(sampled_Y_list)}"
                    except:
                        print("prediction values parsing error!!!")
                        continue
                    
                    eval_predictions = sum(eval_predictions, [])
                    df = pd.DataFrame({f"{property} org labels": org_Y_list, f"{property} sampled labels": sampled_Y_list, "predictions": eval_predictions, "norm_probs": norm_prob_list})
                    df.to_csv(predict_all_file, index = False)

                    R2 = r2_score(sampled_Y_list, eval_predictions)
                    RMSE = mean_squared_error(sampled_Y_list, eval_predictions) ** 0.5
                    MAE = mean_absolute_error(sampled_Y_list, eval_predictions)
                
                    print(R2, RMSE, MAE)
                    with open(performance_file, 'w') as f_out:
                        f_out.write("proxy model checkpoint,test R2,test RMSE,test MAE\n")
                        f_out.write(f"{lora_weights},{R2},{RMSE},{MAE}\n")

    ##################################### start "aggregate with random init. models"
    for lora_weights in lora_weights_list:
        for i in range(exp_cfg.iterations):
            explorer_folder = get_explorer_folder(lora_weights)
            # print(properties)
            eval_folder = os.path.join(explorer_folder, "eval_by_pred_model")
            sub_eval_folders = [os.path.join(eval_folder, x) for x in os.listdir(eval_folder) if os.path.isdir(os.path.join(eval_folder, x))] 
            list.sort(sub_eval_folders)
            
            for property in properties:
                summary_file = os.path.join(eval_folder, property + f"_ensemble_predict_all_iter{i}.csv")
                highest_prob_file = os.path.join(eval_folder, property + f"_best_prob_predict_all_iter{i}.csv")
                
                results_for_ensemble = []
                ensemble_size = len(sub_eval_folders)
                agg = []
                for sub_eval_folder in sub_eval_folders:
                    try:
                        predict_all_file = os.path.join(sub_eval_folder, property + f"_predict_all_iter{i}.csv")
                        df = pd.read_csv(predict_all_file)
                        results_for_ensemble.append(df)
                        agg.append(df["predictions"])
                    except:
                        ensemble_size -= 1
                        agg.append("")
                        continue

                for j in range(0, ensemble_size - 1):
                    if f"{property} org labels" in results_for_ensemble[j].columns:
                        assert results_for_ensemble[-1][f"{property} org labels"].equals(results_for_ensemble[j][f"{property} org labels"])


                try:
                    df_concat = pd.concat(results_for_ensemble)
                except:
                    continue
                df_mean = df_concat.groupby(df_concat.index).mean()    

                df_std =  df_concat.groupby(df_concat.index).std()
                df_std = df_std.loc[:, (df_std != 0).any(axis=0)]
                df_std.columns = [col + '_std' for col in df_std.columns]

                df_summary = pd.concat([df_mean, df_std], axis = 1)
                for j, sub_eval_folder in enumerate(sub_eval_folders):
                    df_summary[sub_eval_folder.split('/')[-1]] = [""] * len(df_summary) if len(agg[j]) == 0 else agg[j]
                agg_drop_empty = [x for x in agg if len(x) > 0]
                if len(agg_drop_empty) > 0:
                    df_pred_concat = pd.concat(agg_drop_empty, axis = 1)

                    df_mean_wo_outliers = df_pred_concat.apply(mean_excluding_one_min_max, axis=1)
                    df_std_wo_outliers = df_pred_concat.apply(std_excluding_one_min_max, axis=1)

                    df_summary["predictions_wo_outliers"] = df_mean_wo_outliers
                    df_summary["predictions_std_wo_outliers"] = df_std_wo_outliers
                    
                df_summary.to_csv(summary_file, index = False)

                eval_Y_list = df_mean[f"{property} sampled labels"]
                eval_predictions = df_mean["predictions"]
                R2 = r2_score(eval_Y_list, eval_predictions)
                RMSE = mean_squared_error(eval_Y_list, eval_predictions) ** 0.5
                MAE = mean_absolute_error(eval_Y_list, eval_predictions)
                
                print(R2, RMSE, MAE)

                scatter_hist(eval_Y_list, eval_predictions, "label", "prediction", title = "ensemble prediction", file_name = os.path.join(eval_folder, property + f"_ensemble_prediction_iter{i}.png"))

                ################# using normalized probs

                for j, df in enumerate(results_for_ensemble):
                    df['origin_index'] = df.index
                    df['source_df'] = j
                combined_df = pd.concat(results_for_ensemble).reset_index(drop=True)
                df_best_probs = combined_df.loc[combined_df.groupby('origin_index')['norm_probs'].idxmax()]
                df_best_probs = df_best_probs.drop(columns=['origin_index', 'source_df'])
                print("df_best_probs:", df_best_probs)
                print(combined_df.groupby('origin_index')['norm_probs'].idxmax())
                df_best_probs.to_csv(highest_prob_file, index = False)

                eval_Y_list = df_best_probs[f"{property} sampled labels"]
                eval_predictions = df_best_probs["predictions"]
                R2_best_probs = r2_score(eval_Y_list, eval_predictions)
                RMSE_best_probs = mean_squared_error(eval_Y_list, eval_predictions) ** 0.5
                MAE_best_probs = mean_absolute_error(eval_Y_list, eval_predictions)

                performance_file = os.path.join(eval_folder, property + "_ensemble_performance.csv")
                with open(performance_file, 'w') as f_out:
                    blank_cell = ''.join("," for _ in range(len(sub_eval_folders)))
                    f_out.write(f"proxy model checkpoint{blank_cell}test R2,test RMSE,test MAE\n")
                    f_out.write(f"{sub_eval_folders},{R2},{RMSE},{MAE}\n")
                    f_out.write(f"selected by normalized probability{blank_cell}{R2_best_probs},{RMSE_best_probs},{MAE_best_probs}\n")

                print(R2_best_probs, RMSE_best_probs, MAE_best_probs)
                scatter_hist(eval_Y_list, eval_predictions, "label", "prediction", title = "best probs. prediction", file_name = os.path.join(eval_folder, property + f"_best_probs_prediction_iter{i}.png"))
        
    #################################### end "aggregate with random init. models"
    ################################# start "aggregate prediction results and recipe "

    for lora_weights in lora_weights_list:
        explorer_folder = get_explorer_folder(lora_weights)
        for i in range(exp_cfg.iterations):
            if os.path.exists(os.path.join(explorer_folder, f"best.json")):
                os.remove(os.path.join(explorer_folder, f"best.json"))
            results_dict = defaultdict(lambda: {"predictions": {k: None for k in exp_cfg.target_props.__dict__}, "predictions_std": {k: None for k in exp_cfg.target_props.__dict__}, \
                                                "original values": {k: None for k in exp_cfg.target_props.__dict__}, \
                                                "target values": {k: None for k in exp_cfg.target_props.__dict__}, "edit distance": None, "changes": None, \
                                                "minimum edit distance": None, "minimum changes": None})

            for prop in exp_cfg.target_props.__dict__:
                eval_folder = os.path.join(explorer_folder, "eval_by_pred_model")
                prediction_results = pd.read_csv(os.path.join(eval_folder, prop + f"_ensemble_predict_all_iter{i}.csv"))

                for j in range(len(prediction_results)):
                    results_dict[j]["predictions"][prop] = prediction_results.loc[j, "predictions_wo_outliers"]
                    results_dict[j]["predictions_std"][prop] = prediction_results.loc[j, "predictions_std_wo_outliers"]
                    results_dict[j]["original values"][prop] = prediction_results.loc[j, f"{prop} org labels"]
                    results_dict[j]["target values"][prop] = prediction_results.loc[j, f"{prop} sampled labels"]
            
            with open(os.path.join(explorer_folder, f"recipe_changes_iter{i}.json"), "r") as f:
                recipe_min_distances = json.load(f)
            for j in range(len(prediction_results)):
                results_dict[j]["edit distance"] = recipe_min_distances[j]

            with open(os.path.join(explorer_folder, f"summary_iter{i}.json"), "w") as f:
                json.dump(results_dict, f, indent = 4)

        best_recipe = []
        for i in range(exp_cfg.iterations):
            with open(os.path.join(explorer_folder, f"summary_iter{i}.json"), "r") as f:
                results_dict = json.load(f)

            for j, (k, v) in enumerate(results_dict.items()):
                if all([v["predictions"][kk] > getattr(getattr(exp_cfg.target_props, kk), "bound")[0] for kk in exp_cfg.target_props.__dict__]) \
                    and not all([v["predictions"][kk] < v["original values"][kk] for kk in exp_cfg.target_props.__dict__]) and v["edit distance"] > 0:
                    best_recipe.append({k: v})

        with open(os.path.join(explorer_folder, f"best.json"), "w") as f:
            json.dump(best_recipe, f, indent = 4)        
    
    

if __name__ == "__main__":
    fire.Fire(main)
