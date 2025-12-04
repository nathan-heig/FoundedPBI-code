#!/usr/bin/env bash
# To be executed as `. run.sh` (or `source run.sh`)

micromamba activate -n pbi-finetune

python finetune_nt2.py  --dataset_base_dir "./finetuning/phage_data/" \
                    --dataset_name "perphect_phage_dataset_12288" \
                    --base_model_name "InstaDeepAI/nucleotide-transformer-v2-50m-multi-species" \
                    --output_model_dir "./finetuning/model_weights/" \
                    --output_model_name "nt2-phages-lora-perphect-lr-3" \
                    --max_length 12288 \
                    --gpu_ids "0" \
                    --num_proc 16 \
                    --lora_rank 8 \
                    --lora_alpha 16 \
                    --lora_dropout 0.1 \
                    --per_device_batch_size 4 \
                    --learning_rate 1e-3 \
                    --epochs 200 \
                    --resume_from_checkpoint
                    