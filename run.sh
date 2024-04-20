#!/bin/bash 

# only quant
#python3 run.py facebook/opt-6.7b c4 braq --blocksize 128 --salient_metric hessian --device "cuda:0"

# 21.609
# python3 run.py /home/dongpeijie/share/llama-2-7b wikitext2 braq --blocksize 128 --salient_metric hessian --device "cuda:0"

# 62.44 
# python3 run.py /data2/share/llama-1/llama-7b-hf wikitext2 braq --blocksize 128 --salient_metric hessian --device "cuda:1"

# prune only: 12.8085
# quant only: 25.778
# quant+prune:977.73
# prune+quant(wanda):155.74
# prune+quant(sparsegpt): 97.01
# prune+quant(sparsegpt)第一层和最后一层不稀疏: 81.97 - 默认操作
# prune+quant(sparsegpt) 前10%和后10% 层不稀疏：97.010


# add pruning
# CUDA_VISIBLE_DEVICES=0 python3 run.py /home/dongpeijie/share/llama-2-7b wikitext2 braq --blocksize 128 \
#     --salient_metric hessian \
#     --device "cuda:0" \
#     --sparsity_ratio 0.5 \
#     --sparsity_type 4:8 \
#     --prune_method wanda

# choices=["xnor", "sign", "no", "2bit", "4bit", "prune", "braq"]
# braq: 81.97
# CUDA_VISIBLE_DEVICES=0 python3 run.py /data2/share/tinyllama/tinyllama-1.1b-480k-1t wikitext2 braq \
#     --blocksize 128 \
#     --salient_metric hessian \
#     --sparsity_ratio 0.5 \
#     --sparsity_type 4:8 \
#     --prune_method sparsegpt

# # xnor 
# CUDA_VISIBLE_DEVICES=0 python3 run.py /data2/share/tinyllama/tinyllama-1.1b-480k-1t wikitext2 xnor \
#     --blocksize 128 \
#     --salient_metric hessian \
#     --sparsity_ratio 0.5 \
#     --sparsity_type 4:8 \
#     --prune_method sparsegpt > ./xnor.log 2>&1 & 

# # sign 
# CUDA_VISIBLE_DEVICES=1 python3 run.py /data2/share/tinyllama/tinyllama-1.1b-480k-1t wikitext2 sign \
#     --blocksize 128 \
#     --salient_metric hessian \
#     --sparsity_ratio 0.5 \
#     --sparsity_type 4:8 \
#     --prune_method sparsegpt > ./sign.log 2>&1 &

# # no
# CUDA_VISIBLE_DEVICES=2 python3 run.py /data2/share/tinyllama/tinyllama-1.1b-480k-1t wikitext2 no \
#     --blocksize 128 \
#     --salient_metric hessian \
#     --sparsity_ratio 0.5 \
#     --sparsity_type 4:8 \
#     --prune_method sparsegpt > ./no.log 2>&1 &

# # 2bit
# CUDA_VISIBLE_DEVICES=3 python3 run.py /data2/share/tinyllama/tinyllama-1.1b-480k-1t wikitext2 2bit \
#     --blocksize 128 \
#     --salient_metric hessian \
#     --sparsity_ratio 0.5 \
#     --sparsity_type 4:8 \
#     --prune_method sparsegpt > ./2bit.log 2>&1 

# # 4bit
# CUDA_VISIBLE_DEVICES=0 python3 run.py /data2/share/tinyllama/tinyllama-1.1b-480k-1t wikitext2 4bit \
#     --blocksize 128 \
#     --salient_metric hessian \
#     --sparsity_ratio 0.5 \
#     --sparsity_type 4:8 \
#     --prune_method sparsegpt > ./4bit.log 2>&1 &

# # prune
# CUDA_VISIBLE_DEVICES=1 python3 run.py /data2/share/tinyllama/tinyllama-1.1b-480k-1t wikitext2 prune \
#     --blocksize 128 \
#     --salient_metric hessian \
#     --sparsity_ratio 0.5 \
#     --sparsity_type 4:8 \
#     --prune_method sparsegpt > ./prune.log 2>&1 &

# baseline 
# CUDA_VISIBLE_DEVICES=1 python3 run.py /data2/share/tinyllama/tinyllama-1.1b-480k-1t wikitext2 prune \
#     --blocksize 128 \
#     --salient_metric hessian \
#     --sparsity_ratio 0.5 \
#     --sparsity_type 4:8 \
#     --prune_method sparsegpt > ./baseline.log 2>&1 &


# wanda 
# CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=2,5 python3 run.py /data2/share/tinyllama/tinyllama-1.1b-480k-1t wikitext2 braq --blocksize 128 \
#     --salient_metric hessian \
#     --sparsity_ratio 0.5 \
#     --sparsity_type 4:8 \
#     --prune_method wanda 

# ria 
CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=4 python3 run.py /data2/share/tinyllama/tinyllama-1.1b-480k-1t wikitext2 braq --blocksize 128 \
    --salient_metric hessian \
    --sparsity_ratio 0.5 \
    --sparsity_type 4:8 \
    --prune_method ria 

# ri 
# CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=5 python3 run.py /data2/share/tinyllama/tinyllama-1.1b-480k-1t wikitext2 braq --blocksize 128 \
#     --salient_metric hessian \
#     --sparsity_ratio 0.5 \
#     --sparsity_type 4:8 \
#     --prune_method ri

# gblm 
# CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=4 python3 run.py /data2/share/tinyllama/tinyllama-1.1b-480k-1t wikitext2 braq --blocksize 128 \
#     --salient_metric hessian \
#     --sparsity_ratio 0.5 \
#     --sparsity_type 4:8 \
#     --prune_method gblm

# 探究那些层对结果影响大；
