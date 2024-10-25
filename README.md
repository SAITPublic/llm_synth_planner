# LLM-Driven Synthesis Planning for Quantum Dot Materials Development
This project focuses on a methodology for identifying experimental procedures for QD materials with desired properties using LLMs.

## Configuration
Before running the scripts, ensure you have correctly set up the configuration file (`config/config_generation.yaml`, `config/config_prediction.yaml`, `config/config_exploration.yaml`). This file should include paths for data and models, as well as other hyperparameters needed for model training.

### Sample Data
The actual data used for training cannot be shared publicly due to data security concerns, as it is internal experimental data from Samsung Electronics. However, we have provided sample data containing QD recipes with the InP composition to help you understand the data format and run the scripts.

## Fine-Tune Models
To fine-tune LLMs with LoRA for the synthesis protocol generation and the property prediction:

```bash
sh finetune_and_generation.sh
```

## Generate and Evaluate Recipes
To generate a new recipe based on desired properties:

```
sh explore_responses.sh
```