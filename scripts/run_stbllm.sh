#!/bin/bash

## llama-2-7b

DIR_PATH=./logs/stbllm

CUDA_VISIBLE_DEVICES=0 python3 run.py /data/lujunli/hf_download/llama-2-7b c4 braq --blocksize 128 \
    --salient_metric hessian \
    --prune_method ria_structure \
    --reconstruction \
    --Lamda 2 \
    --Hyper_m 6 \
    --sparsity_ratio 0.5 \
    --sparsity_type 4:8 > $DIR_PATH/stbllm_base_ria_structure_quant_hessian_billm_llama-2-7b_4_8.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 python3 run.py /data/lujunli/hf_download/llama-2-7b c4 braq --blocksize 128 \
    --salient_metric hessian \
    --prune_method ria_structure \
    --reconstruction \
    --Lamda 2 \
    --Hyper_m 6 \
    --sparsity_ratio 0.5 \
    --sparsity_type 3:8 > $DIR_PATH/stbllm_base_ria_structure_quant_hessian_billm_llama-2-7b_3_8.log 2>&1 &

CUDA_VISIBLE_DEVICES=2 python3 run.py /data/lujunli/hf_download/llama-2-7b c4 braq --blocksize 128 \
    --salient_metric hessian \
    --prune_method ria_structure \
    --reconstruction \
    --Lamda 2 \
    --Hyper_m 6 \
    --sparsity_ratio 0.5 \
    --sparsity_type 2:8 > $DIR_PATH/stbllm_base_ria_structure_quant_hessian_billm_llama-2-7b_2_8.log 2>&1 & 

## llama-1-7b 

CUDA_VISIBLE_DEVICES=3 python3 run.py /data/lujunli/hf_download/llama-1-7b c4 braq --blocksize 128 \
    --salient_metric hessian \
    --prune_method ria_structure \
    --reconstruction \
    --Lamda 2 \
    --Hyper_m 6 \
    --sparsity_ratio 0.5 \
    --sparsity_type 4:8 > $DIR_PATH/stbllm_base_ria_structure_quant_hessian_billm_llama-1-7b_4_8.log 2>&1 & 

CUDA_VISIBLE_DEVICES=4 python3 run.py /data/lujunli/hf_download/llama-1-7b c4 braq --blocksize 128 \
    --salient_metric hessian \
    --prune_method ria_structure \
    --reconstruction \
    --Lamda 2 \
    --Hyper_m 6 \
    --sparsity_ratio 0.5 \
    --sparsity_type 3:8 > $DIR_PATH/stbllm_base_ria_structure_quant_hessian_billm_llama-1-7b_3_8.log 2>&1 &

CUDA_VISIBLE_DEVICES=5 python3 run.py /data/lujunli/hf_download/llama-1-7b c4 braq --blocksize 128 \
    --salient_metric hessian \
    --prune_method ria_structure \
    --reconstruction \
    --Lamda 2 \
    --Hyper_m 6 \
    --sparsity_ratio 0.5 \
    --sparsity_type 2:8 > $DIR_PATH/stbllm_base_ria_structure_quant_hessian_billm_llama-1-7b_2_8.log 2>&1 &
    