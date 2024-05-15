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
# prune+quant(sparsegpt) 第一层和最后一层不稀疏: 81.97 - 默认操作
# prune+quant(sparsegpt) 前10%和后10% 层不稀疏：97.010
# prune+quant(sparsegpt) 第一层和最后一层不稀疏+不对mlp稀疏: 32.84
# prune+quant(sparsegpt) 第一层和最后一层不稀疏+不对attn稀疏: 62.95
# prune+quant(ri): 177451 
# prune+quant(ria): 177451
# prune+quant(sparsegpt)第一block和最后一block不稀疏: 81.97
# prune+quant(sparsegpt)前三block和最后三block不稀疏: 65.326210
# prune+quant(sparsegpt)前25%block和最后25% block不稀疏: 97.01
# prune+quant(sparsegpt)第1层的attn和最后一层的mlp不稀疏: 104.9167
# prune+quant(sparsegpt)第1层的mlp和最后一层的mlp不稀疏: 80.8447
# prune(pruner-zero): 12.63 
# prune+quant(pruner-zero): 151.228
# prune+quant(sparsegpt) first3 last3 mlp 不稀疏：60.278397
# prune+quant(ria+reconstruction): 73.682800
# prune+quant(ria+wo_reconstruction): 73.682800
# prune+quant(ria+reallocation): 91.938477

# prune(ria baseline): 8.100694
# prune(ria only wo reallocation): 7.895372
# prune(ria only w/ reallocation): 8.100694

# add pruning
# CUDA_VISIBLE_DEVICES=1 python3 run.py /data/lujunli/hf_download/llama-2-7b wikitext2 braq --blocksize 128 \
#     --salient_metric hessian \
#     --sparsity_ratio 0.5 \
#     --sparsity_type 4:8 \
#     --prune_method ria \
#     --reconstruction \
#     --reallocation \
#     --lsa > ./logs/ria_reconstruction_reallocation.log 2>&1 &

# CUDA_VISIBLE_DEVICES=5 python3 run.py /data/lujunli/hf_download/llama-2-7b wikitext2 braq --blocksize 128 \
#     --salient_metric hessian \
#     --sparsity_ratio 0.5 \
#     --sparsity_type 4:8 \
#     --prune_method ria \
#     --reconstruction > ./logs/ria_reconstruction_wo_reallocation.log 2>&1 &


# CUDA_VISIBLE_DEVICES=4 python3 run.py /data/lujunli/hf_download/llama-2-7b wikitext2 braq --blocksize 128 \
#     --salient_metric hessian \
#     --sparsity_ratio 0.5 \
#     --sparsity_type 4:8 \
#     --prune_method ria > ./logs/ria_wo_reconstruction_wo_reallocation.log 2>&1 &

# CUDA_VISIBLE_DEVICES=5 python3 run.py /data/lujunli/hf_download/llama-2-7b wikitext2 braq --blocksize 128 \
#     --salient_metric hessian \
#     --sparsity_ratio 0.5 \
#     --sparsity_type 4:8 \
#     --prune_method ria \
#     --reallocation \
#     --lsa > ./logs/ria_wo_reconstruction_w_reallocation.log 2>&1 &

# tail -f ./logs/ria_wo_reconstruction_w_reallocation.log

# CUDA_VISIBLE_DEVICES=5 python3 run.py /data/lujunli/hf_download/llama-2-7b wikitext2 braq --blocksize 128 \
#     --salient_metric hessian \
#     --sparsity_ratio 0.5 \
#     --sparsity_type 4:8 \
#     --prune_method ria \
#     --reconstruction > ./logs/weight_norm_ria_reconstruction_wo_reallocation.log 2>&1 &

# SOTA ppl=56.353821
# CUDA_VISIBLE_DEVICES=5 python3 run.py /data/lujunli/hf_download/llama-2-7b wikitext2 braq --blocksize 128 \
#     --salient_metric hessian \
#     --sparsity_ratio 0.5 \
#     --sparsity_type 4:8 \
#     --Lamda 2 \
#     --Hyper_m 6 \
#     --prune_method ria_structure \
#     --reconstruction > ./logs/ria_structure_reconstruction_wo_reallocation.log 2>&1 &



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

# ria ： 177451
# CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=4 python3 run.py /data2/share/tinyllama/tinyllama-1.1b-480k-1t wikitext2 braq --blocksize 128 \
#     --salient_metric hessian \
#     --sparsity_ratio 0.5 \
#     --sparsity_type 4:8 \
#     --prune_method ria > ./logs/ria.log 2>&1 &

# # ri 
# CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=5 python3 run.py /data2/share/tinyllama/tinyllama-1.1b-480k-1t wikitext2 braq --blocksize 128 \
#     --salient_metric hessian \
#     --sparsity_ratio 0.5 \
#     --sparsity_type 4:8 \
#     --prune_method ri > ./logs/ri.log 2>&1 &

# gblm 
# CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=4 python3 run.py /data2/share/tinyllama/tinyllama-1.1b-480k-1t wikitext2 braq --blocksize 128 \
#     --salient_metric hessian \
#     --sparsity_ratio 0.5 \
#     --sparsity_type 4:8 \
#     --prune_method gblm

# 探究那些层对结果影响大；
# braq: 81.97
# remove mlp: 32.8461
# remove atten: 62.95
# CUDA_VISIBLE_DEVICES=6 python3 run.py /data2/share/tinyllama/tinyllama-1.1b-480k-1t wikitext2 braq \
#     --blocksize 128 \
#     --salient_metric hessian \
#     --sparsity_ratio 0.5 \
#     --sparsity_type 4:8 \
#     --prune_method sparsegpt


# llama7b
# CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=1 python3 run.py /data/lujunli/hf_download/llama-2-7b \
#     wikitext2 braq \
#     --blocksize 128 \
#     --salient_metric hessian \
#     --sparsity_ratio 0.5 \
#     --sparsity_type 4:8 \
#     --prune_method sparsegpt 



# # llama8b
# CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=0,1 python3 run.py /data/lujunli/hf_download/llama-3-8b \
#     wikitext2 braq \
#     --blocksize 128 \
#     --salient_metric hessian \
#     --sparsity_ratio 0.5 \
#     --sparsity_type 4:8 \
#     --prune_method sparsegpt 

# billm 
# ppl = 200
# CUDA_VISIBLE_DEVICES=5 python3 run.py /data/lujunli/hf_download/llama-2-7b wikitext2 braq --blocksize 128 \
#     --salient_metric hessian \
#     --sparsity_ratio 0.5 \
#     --sparsity_type 4:8 \
#     --Lamda 2 \
#     --Hyper_m 6 \
#     --billm \
#     --prune_method ria_structure \
#     --reconstruction
    #  > ./logs/ria_structure_reconstruction_w_billmquant.log 2>&1 &

# pbllm 
# CUDA_VISIBLE_DEVICES=6 python3 run.py /data/lujunli/hf_download/llama-2-7b wikitext2 braq --blocksize 128 \
#     --salient_metric hessian \
#     --sparsity_ratio 0.5 \
#     --sparsity_type 4:8 \
#     --Lamda 2 \
#     --Hyper_m 6 \
#     --pbllm \
#     --prune_method ria_structure \
#     --reconstruction 
    # > ./logs/ria_structure_reconstruction_w_pbllmquant.log 2>&1 &

# gptq 
# 4 bit ppl=8
# 2 bit ppl=186
# CUDA_VISIBLE_DEVICES=7 python3 run.py /data/lujunli/hf_download/llama-2-7b wikitext2 braq --blocksize 128 \
#     --salient_metric hessian \
#     --sparsity_ratio 0.5 \
#     --sparsity_type 4:8 \
#     --Lamda 2 \
#     --Hyper_m 6 \
#     --gptq \
#     --prune_method ria_structure \
#     --reconstruction \
#     --wbits 2 

    # > ./logs/ria_structure_reconstruction_w_gptqquant.log 2>&1 &

# tail -f ./logs/ria_structure_reconstruction_w_billmquant.log


# 2bit 49.682358
CUDA_VISIBLE_DEVICES=7 python3 run.py /data/lujunli/hf_download/tinyllama-1b wikitext2 2bit --blocksize 128 \
    --salient_metric hessian \
    --sparsity_ratio 0.5 \
    --sparsity_type 4:8 \
    --Lamda 2 \
    --Hyper_m 6 \
    --prune_method ria_structure \
    --reconstruction \
    --percdamp 0.04 \
    --high_bit 2 > ./logs/ria_structure_reconstruction_pbllm_2bit.log 2>&1 &

tail -f ./logs/ria_structure_reconstruction_pbllm_2bit.log