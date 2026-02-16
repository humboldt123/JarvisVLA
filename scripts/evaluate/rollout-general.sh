#!/bin/bash

base_url=http://localhost:8000/v1
workers=5
max_frames=500
temperature=0.6
history_num=2
action_chunk_len=1
instruction_type="normal"
model_local_path="CraftJarvis/JarvisVLA-Qwen2-VL-7B"

# Require first argument as task
if [ -z "$1" ]; then
    echo "Usage: $0 <env_config>"
    exit 1
fi

tasks=("$1")

echo "Running for checkpoint $model_local_path..."

for task in "${tasks[@]}"; do
    env_config="$task"

    num_iterations=$(($workers / 5 + 1))
    for ((i = 0; i < num_iterations; i++)); do
        python jarvisvla/evaluate/evaluate.py \
            --workers $workers \
            --env-config $env_config \
            --max-frames $max_frames \
            --temperature $temperature \
            --checkpoints $model_local_path \
            --video-main-fold "logs/" \
            --base-url "$base_url" \
            --history-num $history_num \
            --instruction-type $instruction_type \
            --action-chunk-len $action_chunk_len

        if [[ $? -eq 0 ]]; then
            echo "Python script succeeded on iteration $i, exiting loop."
            break
        fi

        if [[ $i -lt $((num_iterations - 1)) ]]; then
            echo "Waiting 10 seconds..."
            sleep 10
        fi
    done
done
