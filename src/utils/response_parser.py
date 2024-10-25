import json
import os
import re

import numpy as np
import editdistance

def get_predictions_labels_search(
    response_file, test_json_file, reg_patterns, numerical=True
):
    predictions = {}
    with open(response_file, "r") as f:
        single_response = ""
        idx = 0
        for l in f.readlines():
            if l.strip() == "</s>" or l.strip() == "<|end_of_text|>":
                try:
                    single_predictions = [
                        re.search(
                            reg_pattern,
                            single_response.strip().replace("</s>", "").replace("<|end_of_text|>", ""),
                        ).group(1)
                        for reg_pattern in reg_patterns
                    ]
                    if numerical:
                        single_predictions = [
                            float(x) for x in single_predictions
                        ]
                    predictions[idx] = single_predictions
                except:
                    #### contains "," case
                    single_predictions = [
                        re.search(
                            reg_pattern,
                            single_response.replace(",", "")
                            .strip()
                            .replace("</s>", "").replace("<|end_of_text|>", ""),
                        ).group(1)
                        for reg_pattern in reg_patterns
                    ]
                    if numerical:
                        single_predictions = [
                            float(x) for x in single_predictions
                        ]
                    predictions[idx] = single_predictions

                    print(single_response)
                single_response = ""
                idx += 1
                continue
            single_response += l

    if test_json_file is None:
        return list(predictions.values()), []
    elif isinstance(test_json_file, list):
        test_data_list = []
        for single_file in test_json_file:
            with open(single_file, "r") as f:
                test_data_list += json.load(f)
    else:
        with open(test_json_file, "r") as f:
            test_data_list = json.load(f)

    idx = 0
    labels = {}
    for x in test_data_list:
        try:
            label = [
                (
                    j,
                    re.search(
                        reg_pattern, x["output"].strip().replace("</s>", "").replace("<|end_of_text|>", "")
                    ).group(1),
                )
                for j, reg_pattern in enumerate(reg_patterns)
            ]
            labels[idx] = label
        except:
            pass
        idx += 1
    if numerical:
        labels = {
            k: [(y[0], float(y[1])) for y in v] for k, v in labels.items()
        }  # labels = [[(y[0], float(y[1])) for y in x] for x in labels]
    filtered_predictions = []
    assert list(predictions.keys()) == list(
        labels.keys()
    ), f"list(predictions.keys(): {list(predictions.keys())} \n list(labels.keys()): {list(labels.keys())}"
    filtered_labels = list(labels.values())
    for idx, label in labels.items():
        filtered_pred = []
        for j, l in label:
            filtered_pred.append(predictions[idx][j])
        filtered_predictions.append(filtered_pred)
    filtered_labels = [[x[1] for x in label] for label in filtered_labels]
    return filtered_predictions, filtered_labels


def get_predictions_labels_split(
    response_file, test_json_file, split_pattern, numerical=True
):
    predictions = []
    with open(response_file, "r") as f:
        for l in f.readlines():
            single_predictions = [
                x.strip()
                for x in re.split(split_pattern, l.strip().replace("</s>", "").replace("<|end_of_text|>", ""))
                if x.strip()
            ]
            if numerical:
                single_predictions = [float(x) for x in single_predictions]
            predictions.append(single_predictions)

    if test_json_file is None:
        return list(predictions.values()), []
    elif isinstance(test_json_file, list):
        test_data_list = []
        for single_file in test_json_file:
            with open(single_file, "r") as f:
                test_data_list += json.load(f)
    else:
        with open(test_json_file, "r") as f:
            test_data_list = json.load(f)

    labels = [
        [y.strip() for y in re.split(split_pattern, x["output"]) if y.strip()]
        for x in test_data_list
    ]
    if numerical:
        labels = [[float(y) for y in x] for x in labels]
    assert len(predictions) == len(
        labels
    ), f"predictions length: {len(predictions)}\n{predictions}\n\nlabels length: {len(labels)}\n{labels}"
    return predictions, labels

def replace_mask_with_predictions(
    response_file,
    test_json_file,
    split_pattern,
):
    # copy get_predictions_labels_split function. Code refactoring is needed
    predictions = []
    with open(response_file, "r") as f:
        for l in f.readlines():
            single_predictions = [
                x.strip()
                for x in re.split(split_pattern, l.strip().replace("</s>", "").replace("<|end_of_text|>", ""))
                if x.strip()
            ]
            predictions.append(single_predictions)

    if test_json_file is None:
        return list(predictions.values()), []
    elif isinstance(test_json_file, list):
        test_data_list = []
        for single_file in test_json_file:
            with open(single_file, "r") as f:
                test_data_list += json.load(f)
    else:
        with open(test_json_file, "r") as f:
            test_data_list = json.load(f)
    labels = [
        [y.strip() for y in re.split(split_pattern, x["output"]) if y.strip()]
        for x in test_data_list
    ]
    inputs = [x["input"] for x in test_data_list]
    assert len(predictions) == len(labels), f"{predictions}\n{labels}"

    sub_inputs = []
    for pred, label, _input in zip(predictions, labels, inputs):
        for i in range(1, len(label) + 1):
            _input = re.sub(split_pattern.replace("(?:\d+)", str(i)), pred[i - 1], _input)
        sub_inputs.append(_input)

    return sub_inputs


def turn_to_sequence(text):
    text = text.split("Quantum Dot Synthesis Procedure:\n")[-1]
    text = re.sub("## Step \d+", "", text)
    text = re.sub("\n", "", text)
    sequence = text.split("•")
    sequence = [x for x in sequence if len(x) > 0]
    return sequence

def calculate_recipe_similarity(
    response_file,
    test_json_file,
    split_pattern,
):
    if isinstance(test_json_file, list):
        test_data_list = []
        for single_file in test_json_file:
            with open(single_file, "r") as f:
                test_data_list += json.load(f)
    else:
        with open(test_json_file, "r") as f:
            test_data_list = json.load(f)
    labels = [
        [y.strip() for y in re.split(split_pattern, x["output"]) if y.strip()]
        for x in test_data_list
    ]
    inputs = [x["input"] for x in test_data_list]

    true_sub_inputs = []
    for label, _input in zip(labels, inputs):
        for i in range(1, len(label) + 1):
            _input = re.sub(split_pattern.replace("(?:\d+)", str(i)), label[i - 1], _input)
        true_sub_inputs.append(_input)

    sub_inputs = replace_mask_with_predictions(response_file, test_json_file, split_pattern)

    true_recipe_seqs, recipe_seqs = [], []

    for true_sub_input, sub_input in zip(true_sub_inputs, sub_inputs):
        true_sub_input = turn_to_sequence(true_sub_input)
        sub_input = turn_to_sequence(sub_input)
        true_recipe_seqs.append(true_sub_input)
        recipe_seqs.append(sub_input)

    min_dists = []
    for recipe_seq in recipe_seqs:
        min_dist = 100000
        for true_recipe_seq in true_recipe_seqs:
            dist = editdistance.eval(recipe_seq, true_recipe_seq)
            if dist < min_dist:
                min_dist = dist
        min_dists.append(min_dist)
    return min_dists