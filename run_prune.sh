# 测试 ria prune only 

CUDA_VISIBLE_DEVICES=7 python3 run.py \
    /data2/share/tinyllama/tinyllama-1.1b-480k-1t wikitext2 braq \
    --blocksize 128 \
    --salient_metric hessian \
    --sparsity_ratio 0.5 \
    --sparsity_type 4:8 \
    --prune_method pruner-zero > ./logs/pruner-zero.log 2>&1 &

tail -f ./logs/pruner-zero.log 