# prune+quant(sparsegpt)前三block和最后三block不稀疏

CUDA_VISIBLE_DEVICES=6 python3 run.py /data2/share/tinyllama/tinyllama-1.1b-480k-1t wikitext2 braq \
    --blocksize 128 \
    --salient_metric hessian \
    --sparsity_ratio 0.5 \
    --sparsity_type 4:8 \
    --prune_method sparsegpt > ./logs/sparsegpt_f_attn_l_mlp.log 2>&1 &