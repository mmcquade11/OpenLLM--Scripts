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

PEFT_ID="predibase/customer_support"
MODEL_FOR_PEFT="hf-causal-experimental"
MODEL_ID="mistralai/Mistral-7B-v0.1"


git clone -b add-agieval https://github.com/dmahan93/lm-evaluation-harness
cd lm-evaluation-harness
pip install -e .


python main.py \
    --model $MODEL_FOR_PEFT \
    --model_args pretrained=$MODEL_ID,peft=$PEFT_ID \
    --tasks agieval_aqua_rat,agieval_logiqa_en,agieval_lsat_ar,agieval_lsat_lr,agieval_lsat_rc,agieval_sat_en,agieval_sat_en_without_passage,agieval_sat_math \
    --device cuda:0 \
    --output_path ./agi_eval.json

benchmark="gpt4all"
python main.py  \
    --model $MODEL_FOR_PEFT \
    --model_args pretrained=$MODEL_ID,peft=$PEFT_ID \
    --tasks hellaswag,openbookqa,winogrande,arc_easy,arc_challenge,boolq,piqa \
    --device cuda:0 \
    --output_path ./${benchmark}.json

benchmark="truthfulqa"
python main.py  \
    --model $MODEL_FOR_PEFT \
    --model_args pretrained=$MODEL_ID,peft=$PEFT_ID \
    --tasks truthfulqa_mc \
    --device cuda:0 \
    --output_path ./${benchmark}.json

benchmark="bigbench"
python main.py \
    --model $MODEL_FOR_PEFT \
    --model_args pretrained=$MODEL_ID,peft=$PEFT_ID \
    --tasks bigbench_causal_judgement,bigbench_date_understanding,bigbench_disambiguation_qa,bigbench_geometric_shapes,bigbench_logical_deduction_five_objects,bigbench_logical_deduction_seven_objects,bigbench_logical_deduction_three_objects,bigbench_movie_recommendation,bigbench_navigate,bigbench_reasoning_about_colored_objects,bigbench_ruin_names,bigbench_salient_translation_error_detection,bigbench_snarks,bigbench_sports_understanding,bigbench_temporal_sequences,bigbench_tracking_shuffled_objects_five_objects,bigbench_tracking_shuffled_objects_seven_objects,bigbench_tracking_shuffled_objects_three_objects \
    --device cuda:0 \
    --output_path ./${benchmark}.json

end=$(date +%s)
    echo "Elapsed Time: $(($end-$start)) seconds"
    
    python ../llm-autoeval/main.py . $(($end-$start))     