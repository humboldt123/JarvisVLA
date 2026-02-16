#!/bin/bash
#
# Describe mode: query the model in text/VLM mode instead of action mode.
# Make sure vLLM is serving first (scripts/inference/serve_vllm.sh or manual).
#
# Three modes:
#   1. Single question:  asks one question about the scene then exits
#   2. Interactive:       opens a prompt where you can keep asking questions
#   3. Auto-describe:    plays the game AND describes the scene every N steps
#

base_url=http://localhost:8000/v1
model_local_path="CraftJarvis/JarvisVLA-Qwen2-VL-7B"
env_config="kill/kill_zombie"

# --- Pick ONE mode below (comment out the others) ---

# MODE 1: Single question
# python jarvisvla/evaluate/describe.py \
#     --base-url "$base_url" \
#     --checkpoints $model_local_path \
#     --env-config $env_config \
#     --video-main-fold "logs/" \
#     --question "Describe what you see in this Minecraft scene."

# MODE 2: Interactive (ask multiple questions, type 'quit' to exit)
# python jarvisvla/evaluate/describe.py \
#     --base-url "$base_url" \
#     --checkpoints $model_local_path \
#     --env-config $env_config \
#     --video-main-fold "logs/" \
#     --interactive

# MODE 3: Auto-describe (play game + describe every N steps)
python jarvisvla/evaluate/describe.py \
    --base-url "$base_url" \
    --checkpoints $model_local_path \
    --env-config $env_config \
    --video-main-fold "logs/" \
    --auto-describe \
    --describe-interval 20 \
    --max-frames 200 \
    --temperature 0.7 \
    --history-num 2
