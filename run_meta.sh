#!/usr/bin/env bash
# To be executed as `. run.sh` (or `source run.sh`)

micromamba activate -n pbi-finetune

echo "Running fine-tuning with pre-trained NT2 models, chekpoints 0 (phages) and 0 (bacteria)"
export PHAGES_MODEL_NAME="nucleotide-transformer-v2-50m-multi-species"
export BACT_MODEL_NAME="nucleotide-transformer-v2-50m-multi-species"
python main.py -c model_configs/nt2_finetuned.yaml

echo "Running fine-tuning with pre-trained NT2 models, chekpoints 10k (phages)"
export PHAGES_MODEL_NAME="./finetuning/model_weights/nt2-phages-lora-perphect-lr-3/checkpoint-10000"
export BACT_MODEL_NAME="nucleotide-transformer-v2-50m-multi-species"
python main.py -c model_configs/nt2_finetuned.yaml

echo "Running fine-tuning with pre-trained NT2 models, chekpoints 25k (phages)"
export PHAGES_MODEL_NAME="./finetuning/model_weights/nt2-phages-lora-perphect-lr-3/checkpoint-25000"
export BACT_MODEL_NAME="nucleotide-transformer-v2-50m-multi-species"
python main.py -c model_configs/nt2_finetuned.yaml

echo "Running fine-tuning with pre-trained NT2 models, chekpoints 40k (phages)"
export PHAGES_MODEL_NAME="./finetuning/model_weights/nt2-phages-lora-perphect-lr-3/checkpoint-40000"
export BACT_MODEL_NAME="nucleotide-transformer-v2-50m-multi-species"
python main.py -c model_configs/nt2_finetuned.yaml

echo "Running fine-tuning with pre-trained NT2 models, chekpoints 53k (phages)"
export PHAGES_MODEL_NAME="./finetuning/model_weights/nt2-phages-lora-perphect-lr-3/checkpoint-53000"
export BACT_MODEL_NAME="nucleotide-transformer-v2-50m-multi-species"
python main.py -c model_configs/nt2_finetuned.yaml

echo "Running fine-tuning with pre-trained NT2 models, chekpoints 50k (bacteria)"
export PHAGES_MODEL_NAME="nucleotide-transformer-v2-50m-multi-species"
export BACT_MODEL_NAME="./finetuning/model_weights/nt2-bact-lora-perphect-lr-3/checkpoint-50000"
python main.py -c model_configs/nt2_finetuned.yaml

echo "Running fine-tuning with pre-trained NT2 models, chekpoints 100k (bacteria)"
export PHAGES_MODEL_NAME="nucleotide-transformer-v2-50m-multi-species"
export BACT_MODEL_NAME="./finetuning/model_weights/nt2-bact-lora-perphect-lr-3/checkpoint-100000"
python main.py -c model_configs/nt2_finetuned.yaml

echo "Running fine-tuning with pre-trained NT2 models, chekpoints 155k (bacteria)"
export PHAGES_MODEL_NAME="nucleotide-transformer-v2-50m-multi-species"
export BACT_MODEL_NAME="./finetuning/model_weights/nt2-bact-lora-perphect-lr-3/checkpoint-155000"
python main.py -c model_configs/nt2_finetuned.yaml