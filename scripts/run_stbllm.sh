#!/bin/bash

# Update baseline from sparsegpt to ria_structure;

LOG_PATH=./logs/stbllm_series

if [ ! -d $LOG_PATH ]; then
    mkdir -p $LOG_PATH
    echo "Created directory: $LOG_PATH"
else
    echo "Directory already exists: $LOG_PATH"
fi

MODEL_NAME_LIST=(
    "llama-2-13b"
    "llama-1-13b"
    "llama-1-33b"
    "llama-3-8b"
    "mistral-7b"
    "llama-2-7b"
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
GPUS=(7) 

# Initialize experiment counter
experiment_count=0

# Array to store the PIDs of background jobs
job_pids=()

# Array to store the PID corresponding to each GPU
gpu_to_pid=()

# Infinite loop to continuously process batches
while true; do
    # Iterate over GPUs to check for availability
    for gpu in "${GPUS[@]}"; do
        # If the GPU is currently not running any experiment or the corresponding process has exited
        if [[ -z "${gpu_to_pid[$gpu]}" ]] || ! kill -0 "${gpu_to_pid[$gpu]}" 2>/dev/null; then
            echo "GPU $gpu is free or job exited, starting a new experiment"
            
            # Select the model and sparsity configuration based on experiment count
            MODEL_NAME=${MODEL_NAME_LIST[$((experiment_count / 3))]}
            SPARSITY_RATIO=${SPARSITY_RATIO_LIST[$((experiment_count % 3))]}
            SPARSITY_TYPE=${SPARSITY_TYPE_LIST[$((experiment_count % 3))]}
            
            echo "Starting experiment $((experiment_count + 1)) on GPU $gpu with Model: $MODEL_NAME, Sparsity ratio: $SPARSITY_RATIO, Sparsity type: $SPARSITY_TYPE"


            CUDA_VISIBLE_DEVICES=$gpu python3 run.py /path/to/hf-model/${MODEL_NAME} c4 braq --blocksize 128 \
                --salient_metric hessian \
                --prune_method ria_structure \
                --reconstruction \
                --Lamda 2 \
                --Hyper_m 6 \
                --sparsity_ratio ${SPARSITY_RATIO} \
                --sparsity_type ${SPARSITY_TYPE} > $LOG_PATH/stbllm_${MODEL_NAME}_c4_ria-structure-${SPARSITY_TYPE}-${SPARSITY_RATIO}-wrec_hessian-4mask-billm-160_main.log 2>&1 &

            
            # Store the PID of the background job
            job_pids+=($!)
            # Update the GPU-to-PID mapping
            gpu_to_pid[$gpu]=$!
            # Increment the experiment count
            ((experiment_count++))

            # Break the loop if all experiments are completed
            if ((experiment_count >= ${#MODEL_NAME_LIST[@]} * ${#SPARSITY_RATIO_LIST[@]})); then
                echo "All experiments completed."
                exit 0
            fi
        fi
    done

    # Wait for a short duration before checking again to avoid excessive CPU usage
    sleep 1
done
