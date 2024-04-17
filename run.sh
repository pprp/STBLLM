#!/bin/bash 

# only quant
#python3 run.py facebook/opt-6.7b c4 braq --blocksize 128 --salient_metric hessian --device "cuda:0"

# 21.609
# python3 run.py /home/dongpeijie/share/llama-2-7b wikitext2 braq --blocksize 128 --salient_metric hessian --device "cuda:0"

# 62.44 
# python3 run.py /data2/share/llama-1/llama-7b-hf wikitext2 braq --blocksize 128 --salient_metric hessian --device "cuda:1"

# add pruning
# CUDA_VISIBLE_DEVICES=0 python3 run.py /home/dongpeijie/share/llama-2-7b wikitext2 braq --blocksize 128 \
#     --salient_metric hessian \
#     --device "cuda:0" \
#     --sparsity_ratio 0.5 \
#     --sparsity_type 4:8 \
#     --prune_method wanda 

CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=0,1,6,7 python3 run.py /home/dongpeijie/share/llama-2-7b wikitext2 braq --blocksize 128 \
    --salient_metric hessian \
    --sparsity_ratio 0.5 \
    --sparsity_type 4:8 \
    --prune_method wanda 