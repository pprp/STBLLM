#!/bin/bash

# Update baseline from sparsegpt to ria_structure;

LOG_PATH=./logs/stbllm_65b_70b

if [ ! -d $LOG_PATH ]; then
    mkdir -p $LOG_PATH
    echo "Created directory: $LOG_PATH"
else
    echo "Directory already exists: $LOG_PATH"
fi

MODEL_NAME_LIST=(
    # "llama-1-65b"
    "llama-2-70b"
)

SPARSITY_RATIO_LIST=(
    0.25
    0.375
    0.5
)

SPARSITY_TYPE_LIST=(
    "2:8"
    "3:8"
    "4:8"
)

# Define GPU array
GPUS=(0 1 2 5) # Assuming we have only five GPUs

# Initialize experiment counter
experiment_count=0

# Array to store the PIDs of background jobs
job_pids=()

# Array to store the PID corresponding to each GPU
gpu_to_pid=()

for MODEL_NAME in "${MODEL_NAME_LIST[@]}"
do
    for i in "${!SPARSITY_RATIO_LIST[@]}"
    do
        SPARSITY_RATIO="${SPARSITY_RATIO_LIST[i]}"
        SPARSITY_TYPE="${SPARSITY_TYPE_LIST[i]}"
        
        CUDA_VISIBLE_DEVICES=0,1,2,4 python3 run.py /aifs4su/mmdata/hf_download/$MODEL_NAME c4 braq --blocksize 128 \
            --salient_metric hessian \
            --prune_method ria_structure \
            --reconstruction \
            --Lamda 2 \
            --Hyper_m 6 \
            --sparsity_ratio $SPARSITY_RATIO \
            --sparsity_type $SPARSITY_TYPE > $LOG_PATH/stbllm_${MODEL_NAME}_c4_ria-structure-${SPARSITY_TYPE}-${SPARSITY_RATIO}-wrec_hessian-4mask-billm-160_main.log 2>&1 &
        
        job_pids+=($!)
        gpu_to_pid[$experiment_count % ${#GPUS[@]}]=$!
        experiment_count=$((experiment_count + 1))
        
        # Wait for the current job to finish before starting the next one
        wait $!
    done
done

echo "All experiments completed."