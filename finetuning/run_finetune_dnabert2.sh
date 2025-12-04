#!/usr/bin/env bash
# To be executed as `. run.sh` (or `source run.sh`)

micromamba activate -n pbi-dnabert

DATA_PATH="/home/pere.carrillo/data-local/pbi/finetuning/datasets/dnabert_phages"  # e.g., ./sample_data
MAX_LENGTH=6000 # Please set the number as 0.25 * your sequence length. 
											# e.g., set it as 250 if your DNA sequences have 1000 nucleotide bases
											# This is because the tokenized will reduce the sequence length by about 5 times
LR=3e-5
export CUDA_VISIBLE_DEVICES=2  # Set GPU IDs

# Training use DataParallel
python ~/data-local/pbi/finetuning/DNABERT_2/finetune/train.py \
    --model_name_or_path ~/data-local/pbi/foundation-models-test/DNABERT-2/DNABERT-2-117M/ \
    --data_path  ${DATA_PATH} \
    --kmer -1 \
    --run_name DNABERT2_${DATA_PATH} \
    --model_max_length ${MAX_LENGTH} \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 16 \
    --gradient_accumulation_steps 1 \
    --learning_rate ${LR} \
    --num_train_epochs 5 \
    --fp16 \
    --save_steps 200 \
    --output_dir output/dnabert2 \
    --evaluation_strategy steps \
    --eval_strategy steps \
    --save_strategy steps \
    --eval_steps 200 \
    --warmup_steps 50 \
    --logging_steps 100 \
    --overwrite_output_dir True \
    --log_level info \
    --find_unused_parameters False