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

# choices=["xnor", "sign", "no", "2bit", "4bit", "prune", "braq"]
# braq: 81.97
# CUDA_VISIBLE_DEVICES=0 python3 run.py /mnt/sdb/dongpeijie/download/TinyLlama/TinyLlama-1.1B-intermediate-step-480k-1T wikitext2 braq \
#     --blocksize 128 \
#     --salient_metric hessian \
#     --sparsity_ratio 0.5 \
#     --sparsity_type 4:8 \
#     --prune_method sparsegpt

# # xnor 
# CUDA_VISIBLE_DEVICES=0 python3 run.py /mnt/sdb/dongpeijie/download/TinyLlama/TinyLlama-1.1B-intermediate-step-480k-1T wikitext2 xnor \
#     --blocksize 128 \
#     --salient_metric hessian \
#     --sparsity_ratio 0.5 \
#     --sparsity_type 4:8 \
#     --prune_method sparsegpt > ./xnor.log 2>&1 & 

# # sign 
# CUDA_VISIBLE_DEVICES=1 python3 run.py /mnt/sdb/dongpeijie/download/TinyLlama/TinyLlama-1.1B-intermediate-step-480k-1T wikitext2 sign \
#     --blocksize 128 \
#     --salient_metric hessian \
#     --sparsity_ratio 0.5 \
#     --sparsity_type 4:8 \
#     --prune_method sparsegpt > ./sign.log 2>&1 &

# # no
# CUDA_VISIBLE_DEVICES=2 python3 run.py /mnt/sdb/dongpeijie/download/TinyLlama/TinyLlama-1.1B-intermediate-step-480k-1T wikitext2 no \
#     --blocksize 128 \
#     --salient_metric hessian \
#     --sparsity_ratio 0.5 \
#     --sparsity_type 4:8 \
#     --prune_method sparsegpt > ./no.log 2>&1 &

# # 2bit
# CUDA_VISIBLE_DEVICES=3 python3 run.py /mnt/sdb/dongpeijie/download/TinyLlama/TinyLlama-1.1B-intermediate-step-480k-1T wikitext2 2bit \
#     --blocksize 128 \
#     --salient_metric hessian \
#     --sparsity_ratio 0.5 \
#     --sparsity_type 4:8 \
#     --prune_method sparsegpt > ./2bit.log 2>&1 

# # 4bit
# CUDA_VISIBLE_DEVICES=0 python3 run.py /mnt/sdb/dongpeijie/download/TinyLlama/TinyLlama-1.1B-intermediate-step-480k-1T wikitext2 4bit \
#     --blocksize 128 \
#     --salient_metric hessian \
#     --sparsity_ratio 0.5 \
#     --sparsity_type 4:8 \
#     --prune_method sparsegpt > ./4bit.log 2>&1 &

# # prune
# CUDA_VISIBLE_DEVICES=1 python3 run.py /mnt/sdb/dongpeijie/download/TinyLlama/TinyLlama-1.1B-intermediate-step-480k-1T wikitext2 prune \
#     --blocksize 128 \
#     --salient_metric hessian \
#     --sparsity_ratio 0.5 \
#     --sparsity_type 4:8 \
#     --prune_method sparsegpt > ./prune.log 2>&1 &

# baseline 
CUDA_VISIBLE_DEVICES=1 python3 run.py /mnt/sdb/dongpeijie/download/TinyLlama/TinyLlama-1.1B-intermediate-step-480k-1T wikitext2 prune \
    --blocksize 128 \
    --salient_metric hessian \
    --sparsity_ratio 0.5 \
    --sparsity_type 4:8 \
    --prune_method sparsegpt > ./baseline.log 2>&1 &



# CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=2,5 python3 run.py /mnt/sdb/dongpeijie/download/TinyLlama/TinyLlama-1.1B-intermediate-step-480k-1T wikitext2 braq --blocksize 128 \
#     --salient_metric hessian \
#     --sparsity_ratio 0.5 \
#     --sparsity_type 4:8 \
#     --prune_method wanda 