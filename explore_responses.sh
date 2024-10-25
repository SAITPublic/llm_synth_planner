#!/bin/bash

dir_path=${PWD}
export PYTHONPATH=$PYTHONPATH:$dir_path

model_config="${dir_path}/config/config_exploration.yaml"
exploration_config="${dir_path}/config/VQY_SF_target_regions.yaml"



python3 ${dir_path}/explore_responses.py \
--model_config ${model_config} \
--exploration_config ${exploration_config} 


