#!/bin/bash

start=$(date +%s)

# Detect the number of NVIDIA GPUs and create a device string
gpu_count=$(nvidia-smi -L | wc -l)
if [ $gpu_count -eq 0 ]; then
    echo "No NVIDIA GPUs detected. Exiting."
    exit 1
fi
# Construct the CUDA device string
cuda_devices=""
for ((i=0; i<gpu_count; i++)); do
    if [ $i -gt 0 ]; then
        cuda_devices+=","
    fi
    cuda_devices+="$i"
done

# Install dependencies
apt update
apt install -y screen vim git-lfs
screen

export AWS_ACCESS_KEY_ID="$ARCEE_AWS_ACCESS_KEY_ID"
export AWS_SECRET_ACCESS_KEY="$ARCEE_AWS_SECRET_ACCESS_KEY"
export AWS_DEFAULT_REGION=us-east-2

echo "aws key:" "$AWS_ACCESS_KEY_ID"

# Install common libraries
pip install -q requests accelerate sentencepiece pytablewriter einops protobuf

MODEL_FOR_OPENLLM="vllm"
MODEL_FOR_OPENLLM_MMLU="hf-causal-experimental"
MODEL_ID="mistralai/Mistral-7B-v0.1"

git clone https://github.com/EleutherAI/lm-evaluation-harness
    cd lm-evaluation-harness
    python -m pip install --upgrade pip
    pip install -e .
    pip install --upgrade vllm
    pip install --upgrade promptsource
    pip install langdetect immutabledict

    benchmark="arc"
    lm_eval --model $MODEL_FOR_OPENLLM \
        --model_args pretrained=${MODEL_ID},trust_remote_code=$TRUST_REMOTE_CODE \
        --tasks arc_challenge \
        --num_fewshot 25 \
        --batch_size auto \
        --output_path ./${benchmark}.json

    benchmark="hellaswag"
    lm_eval --model $MODEL_FOR_OPENLLM \
        --model_args pretrained=${MODEL_ID},trust_remote_code=$TRUST_REMOTE_CODE \
        --tasks hellaswag \
        --num_fewshot 10 \
        --batch_size auto \
        --output_path ./${benchmark}.json

    benchmark="mmlu"
    lm_eval --model $MODEL_FOR_OPENLLM_MMLU \
        --model_args pretrained=${MODEL_ID},trust_remote_code=$TRUST_REMOTE_CODE \
        --tasks mmlu \
        --num_fewshot 5 \
        --batch_size auto \
        --verbosity DEBUG \
        --output_path ./${benchmark}.json
    
    benchmark="truthfulqa"
    lm_eval --model $MODEL_FOR_OPENLLM \
        --model_args pretrained=${MODEL_ID},trust_remote_code=$TRUST_REMOTE_CODE \
        --tasks truthfulqa \
        --num_fewshot 0 \
        --batch_size auto \
        --output_path ./${benchmark}.json
    
    benchmark="winogrande"
    lm_eval --model $MODEL_FOR_OPENLLM \
        --model_args pretrained=${MODEL_ID},trust_remote_code=$TRUST_REMOTE_CODE \
        --tasks winogrande \
        --num_fewshot 5 \
        --batch_size auto \
        --output_path ./${benchmark}.json
    
    benchmark="gsm8k"
    lm_eval --model $MODEL_FOR_OPENLLM \
        --model_args pretrained=${MODEL_ID},trust_remote_code=$TRUST_REMOTE_CODE \
        --tasks gsm8k \
        --num_fewshot 5 \
        --batch_size auto \
        --output_path ./${benchmark}.json

    end=$(date +%s)
    echo "Elapsed Time: $(($end-$start)) seconds"
    
    python ../llm-autoeval/main.py . $(($end-$start))

