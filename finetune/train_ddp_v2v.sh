#!/usr/bin/env bash

# Prevent tokenizer parallelism issues
export TOKENIZERS_PARALLELISM=false

# Model Configuration # checkpoint and deepspeed to reduce the VRAM usage
MODEL_ARGS=(
    --model_path "THUDM/CogVideoX-5B"
    --model_name "cogvideox-i2v"  # ["cogvideox-i2v"]
    --model_type "i2v" #v2v? or i2v? # if V2V can use CogVideoX-5B
    --training_type "lora"
)

# Output Configuration
OUTPUT_ARGS=(
    --output_dir "/home/kmwuab/CogVideo/CogVideo_CheckPoint_V2V"
    --report_to "tensorboard"
)

# Data Configuration
DATA_ARGS=(
    --data_root "/home/kmwuab/CogVideo/CogVideo_dataset/360x_dataset_HR/cogvideo_dataset"
    --caption_column "/home/kmwuab/CogVideo/CogVideo_dataset/360x_dataset_HR/cogvideo_dataset/prompts.txt"
    --input_video_column "/home/kmwuab/CogVideo/CogVideo_dataset/360x_dataset_HR/cogvideo_dataset/input_videos_mp4.txt"
    --gt_video_column "/home/kmwuab/CogVideo/CogVideo_dataset/360x_dataset_HR/cogvideo_dataset/target_videos_mp4.txt"
    --video_column "/home/kmwuab/CogVideo/CogVideo_dataset/360x_dataset_HR/cogvideo_dataset/target_videos_mp4.txt"
    --train_resolution "49x480x720"  # (frames x height x width), frames should be 8N+1 and height, width should be multiples of 16
)

# Training Configuration
TRAIN_ARGS=(
    --train_epochs 25 # number of training epochs
    --seed 42 # random seed
    --batch_size 1 # previous 32
    --gradient_accumulation_steps 5 # previous 1 # related to checkpoint
    --mixed_precision "bf16"  # ["no", "fp16"] # Only CogVideoX-2B supports fp16 training
)

# System Configuration
SYSTEM_ARGS=(
    --num_workers 4
    --pin_memory True
    --nccl_timeout 180 #1800?
)

# Checkpointing Configuration
CHECKPOINT_ARGS=(
    --checkpointing_steps 10 # save checkpoint every x steps
    --checkpointing_limit 2 # maximum number of checkpoints to keep, after which the oldest one is deleted
    #--resume_from_checkpoint "/disk1/jinmin/CogVideo_CheckPoint"  # if you want to resume from a checkpoint, otherwise, comment this line
)

# Validation Configuration
VALIDATION_ARGS=(
    --do_validation false  # ["true", "false"]
    --validation_dir "/absolute/path/to/your/validation_set"
    --validation_steps 20  # should be multiple of checkpointing_steps
    --validation_prompts "prompts.txt"
    --validation_images "images.txt"
    --gen_fps 16
)

# Combine all arguments and launch training # original using accelerate/ deepspeed
accelerate launch --num_processes=1 /home/kmwuab/CogVideo/finetune/train_new.py \
    "${MODEL_ARGS[@]}" \
    "${OUTPUT_ARGS[@]}" \
    "${DATA_ARGS[@]}" \
    "${TRAIN_ARGS[@]}" \
    "${SYSTEM_ARGS[@]}" \
    "${CHECKPOINT_ARGS[@]}" \
    "${VALIDATION_ARGS[@]}"