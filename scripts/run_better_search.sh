#!/bin/bash

# 定义模型路径
MODEL_PATH="/home/lujunli/hf_download/llama-2-7b"

# 定义GPU数组
GPUS=(0 2 3 4 5) # 假设我们只有两个GPU

# 初始化实验计数器
experiment_count=180

# 用于存储后台作业PID的数组
job_pids=()

# 用于存储每个GPU对应的PID
gpu_to_pid=()

# 无限循环，用于连续处理批次
while true; do
    # 检查是否有GPU空闲
    for gpu in "${GPUS[@]}"; do
        # 如果这个GPU当前没有运行任何实验，或者对应的进程已经退出
        if [[ -z "${gpu_to_pid[$gpu]}" ]] || ! kill -0 "${gpu_to_pid[$gpu]}" 2>/dev/null; then
            echo "GPU $gpu is free or job exited, starting a new experiment"
            
            # 如果实验计数器大于等于当前job_pids数组的长度，说明需要启动新的实验
            if [[ $experiment_count -ge ${#job_pids[@]} ]]; then
                echo "Starting experiment ${experiment_count} on GPU $gpu"
                CUDA_VISIBLE_DEVICES=$gpu python3 run.py /data/lujunli/hf_download/llama-2-7b c4 braq --blocksize 128 \
                    --salient_metric auto \
                    --prune_method ria_structure \
                    --reconstruction \
                    --Lamda 2 \
                    --Hyper_m 6 \
                    --sparsity_ratio 0.5 \
                    --sparsity_type 4:8 > "./logs/search/search_quant_metric_exp${experiment_count}_gpu$gpu.log" 2>&1 &
                job_pids+=($!) # 保存后台作业的PID
                gpu_to_pid[$gpu]=$! # 更新GPU到PID的映射
                ((experiment_count++)) # 递增实验计数器
            fi
        fi
    done

    # 等待一段时间再进行下一次检查，避免过度占用CPU资源
    sleep 1
done