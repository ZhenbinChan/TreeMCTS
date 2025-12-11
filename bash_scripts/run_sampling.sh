#!/bin/bash

set -x

cd /home/chenzhb/Workspaces/TreeMCTS

python src/main.py \
    --model_id /home/chenzhb/Workspaces/LLMs/Qwen2.5-7B-Instruct \
    --dataset_id olegbask/AR-LSAT \
    --num_samples 2 \
    --num_branches 2 \
    --top_k_entropy 2 \
    --output_dir ./outputs \
    --seed 42 
