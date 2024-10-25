#!/bin/bash

dir_path=${PWD}
export PYTHONPATH=$PYTHONPATH:$dir_path

# ################ train for generation
# model_config=${dir_path}/config/config_generation.yaml
# python finetune.py $model_config
# python generate.py $model_config

################ train for prediction
model_config=${dir_path}/config/config_prediction.yaml
dataloader_config=${dir_path}/config/add_noise.yaml
python finetune.py $model_config $dataloader_config
python generate.py $model_config $dataloader_config