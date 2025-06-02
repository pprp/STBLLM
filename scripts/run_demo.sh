#!/bin/bash

# Update baseline from sparsegpt to si_structure;

LOG_PATH=./logs/stbllm_series

if [ ! -d $LOG_PATH ]; then
    mkdir -p $LOG_PATH
    echo "Created directory: $LOG_PATH"
else
    echo "Directory already exists: $LOG_PATH"
fi

MODEL_NAME="/data2/share/llama-2/Llama-2-7b-hf"
SPARSITY_RATIO=0.5
SPARSITY_TYPE="4:8"

# Define GPU array
GPUS=(3)

echo "Starting experiment on GPU ${GPUS[0]} with Model: $MODEL_NAME, Sparsity ratio: $SPARSITY_RATIO, Sparsity type: $SPARSITY_TYPE"

time_start=$(date +%s)
CUDA_VISIBLE_DEVICES=${GPUS[0]} python3 run.py ${MODEL_NAME} wikitext2 braq --blocksize 128 \
    --salient_metric hessian \
    --prune_method si_structure \
    --reconstruction \
    --Lamda 2 \
    --Hyper_m 6 \
    --sparsity_ratio ${SPARSITY_RATIO} \
    --sparsity_type ${SPARSITY_TYPE} > $LOG_PATH/stbllm_wikitext2_si-structure-${SPARSITY_TYPE}-${SPARSITY_RATIO}-wrec_hessian-4mask-billm-160_main.log 2>&1

time_end=$(date +%s)
time_cost=$((time_end - time_start))
echo "Time cost: $time_cost seconds"

# Demo
# prune 7.63
# quant 9.38
# prune+quant 20.6