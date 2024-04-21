# prune+quant(sparsegpt)前三block和最后三block不稀疏

# sparsegpt

# CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=7 python3 run.py /data2/share/tinyllama/tinyllama-1.1b-480k-1t \
#     wikitext2 braq \
#     --blocksize 128 \
#     --salient_metric hessian \
#     --sparsity_ratio 0.5 \
#     --sparsity_type 4:8 \
#     --prune_method sparsegpt > ./logs/sparsegpt_f_4_l_4_bs128.log 2>&1 

# CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=7 python3 run.py /data2/share/tinyllama/tinyllama-1.1b-480k-1t \
#     wikitext2 braq \
#     --blocksize 64 \
#     --salient_metric hessian \
#     --sparsity_ratio 0.5 \
#     --sparsity_type 4:8 \
#     --prune_method sparsegpt > ./logs/sparsegpt_f_4_l_4_bs64.log 2>&1 

# CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=7 python3 run.py /data2/share/tinyllama/tinyllama-1.1b-480k-1t \
#     wikitext2 braq \
#     --blocksize 32 \
#     --salient_metric hessian \
#     --sparsity_ratio 0.5 \
#     --sparsity_type 4:8 \
#     --prune_method sparsegpt > ./logs/sparsegpt_f_4_l_4_bs32.log 2>&1

# CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=7 python3 run.py /data2/share/tinyllama/tinyllama-1.1b-480k-1t \
#     wikitext2 braq \
#     --blocksize 16 \
#     --salient_metric hessian \
#     --sparsity_ratio 0.5 \
#     --sparsity_type 4:8 \
#     --prune_method sparsegpt > ./logs/sparsegpt_f_4_l_4_bs16.log 2>&1

# # wanda 
# CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=7 python3 run.py /data2/share/tinyllama/tinyllama-1.1b-480k-1t \
#     wikitext2 braq \
#     --blocksize 128 \
#     --salient_metric hessian \
#     --sparsity_ratio 0.5 \
#     --sparsity_type 4:8 \
#     --prune_method wanda > ./logs/wanda_f_4_l_4_bs128.log 2>&1

# CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=7 python3 run.py /data2/share/tinyllama/tinyllama-1.1b-480k-1t \
#     wikitext2 braq \
#     --blocksize 64 \
#     --salient_metric hessian \
#     --sparsity_ratio 0.5 \
#     --sparsity_type 4:8 \
#     --prune_method wanda > ./logs/wanda_f_4_l_4_bs64.log 2>&1

# CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=7 python3 run.py /data2/share/tinyllama/tinyllama-1.1b-480k-1t \
#     wikitext2 braq \
#     --blocksize 32 \
#     --salient_metric hessian \
#     --sparsity_ratio 0.5 \
#     --sparsity_type 4:8 \
#     --prune_method wanda > ./logs/wanda_f_4_l_4_bs32.log 2>&1

# CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=7 python3 run.py /data2/share/tinyllama/tinyllama-1.1b-480k-1t \
#     wikitext2 braq \
#     --blocksize 16 \
#     --salient_metric hessian \
#     --sparsity_ratio 0.5 \
#     --sparsity_type 4:8 \
#     --prune_method wanda > ./logs/wanda_f_4_l_4_bs16.log 2>&1

# # Quant Range (origin is -1, 1000)
# CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=7 python3 run.py /data2/share/tinyllama/tinyllama-1.1b-480k-1t \
#     wikitext2 braq \
#     --blocksize 128 \
#     --salient_metric hessian \
#     --sparsity_ratio 0.5 \
#     --sparsity_type 4:8 \
#     --prune_method sparsegpt \
#     --minlayer 1 \
#     --maxlayer 19 > ./logs/sparsegpt_f_4_l_4_bs128_mn1_mx19.log 2>&1

# CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=7 python3 run.py /data2/share/tinyllama/tinyllama-1.1b-480k-1t \
#     wikitext2 braq \
#     --blocksize 128 \
#     --salient_metric hessian \
#     --sparsity_ratio 0.5 \
#     --sparsity_type 4:8 \
#     --prune_method sparsegpt \
#     --minlayer 2 \
#     --maxlayer 18 > ./logs/sparsegpt_f_4_l_4_bs128_mn2_mx18.log 2>&1

# CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=7 python3 run.py /data2/share/tinyllama/tinyllama-1.1b-480k-1t \
#     wikitext2 braq \
#     --blocksize 128 \
#     --salient_metric hessian \
#     --sparsity_ratio 0.5 \
#     --sparsity_type 4:8 \
#     --prune_method sparsegpt \
#     --minlayer 3 \
#     --maxlayer 17 > ./logs/sparsegpt_f_4_l_4_bs128_mn3_mx17.log 2>&1

# CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=7 python3 run.py /data2/share/tinyllama/tinyllama-1.1b-480k-1t \
#     wikitext2 braq \
#     --blocksize 128 \
#     --salient_metric hessian \
#     --sparsity_ratio 0.5 \
#     --sparsity_type 4:8 \
#     --prune_method sparsegpt \
#     --minlayer 4 \
#     --maxlayer 16 > ./logs/sparsegpt_f_4_l_4_bs128_mn4_mx16.log 2>&1

# CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=7 python3 run.py /data2/share/tinyllama/tinyllama-1.1b-480k-1t \
#     wikitext2 braq \
#     --blocksize 128 \
#     --salient_metric hessian \
#     --sparsity_ratio 0.5 \
#     --sparsity_type 4:8 \
#     --prune_method sparsegpt \
#     --minlayer 5 \
#     --maxlayer 15 > ./logs/sparsegpt_f_4_l_4_bs128_mn5_mx15.log 2>&1

CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=2,3 python3 run.py /data/lujunli/hf_download/llama-2-13b \
    wikitext2 braq \
    --blocksize 128 \
    --salient_metric hessian \
    --sparsity_ratio 0.5 \
    --sparsity_type 4:8 \
    --prune_method sparsegpt 