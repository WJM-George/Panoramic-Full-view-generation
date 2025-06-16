#!/usr/bin/env bash

# Prevent tokenizer parallelism issues
export TOKENIZERS_PARALLELISM=false

# Model Configuration
MODEL_ARGS=(
    --model_path "THUDM/CogVideoX-5B-I2V"
    --model_name "cogvideox-i2v"  # ["cogvideox-i2v"]
    --model_type "i2v"
    --training_type "sft"
)

# Output Configuration
OUTPUT_ARGS=(
    --output_dir "/home/kmwuab/CogVideo/CogVideo_CheckPoint"
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
    --train_epochs 10 # number of training epochs
    --seed 42 # random seed

    #########   Please keep consistent with deepspeed config file ##########
    --batch_size 16
    --gradient_accumulation_steps 5
    --mixed_precision "bf16"  # ["no", "fp16"] Only CogVideoX-2B supports fp16 training
    ########################################################################
)

# System Configuration
SYSTEM_ARGS=(
    --num_workers 4
    --pin_memory True
    --nccl_timeout 180
)

# Checkpointing Configuration
CHECKPOINT_ARGS=(
    --checkpointing_steps 20 # save checkpoint every x steps
    --checkpointing_limit 5 # maximum number of checkpoints to keep, after which the oldest one is deleted
    # --resume_from_checkpoint "/absolute/path/to/checkpoint_dir"  # if you want to resume from a checkpoint, otherwise, comment this line
)

# Validation Configuration
VALIDATION_ARGS=(
    --do_validation false  # ["true", "false"]
    --validation_dir "/absolute/path/to/validation_set"
    --validation_steps 20  # should be multiple of checkpointing_steps
    --validation_prompts "prompts.txt"
    --validation_images "images.txt"
    --gen_fps 16
)

# Combine all arguments and launch training
accelerate launch --config_file /home/kmwuab/CogVideo/finetune/accelerate_config.yaml /home/kmwuab/CogVideo/finetune/train.py \
    "${MODEL_ARGS[@]}" \
    "${OUTPUT_ARGS[@]}" \
    "${DATA_ARGS[@]}" \
    "${TRAIN_ARGS[@]}" \
    "${SYSTEM_ARGS[@]}" \
    "${CHECKPOINT_ARGS[@]}" \
    "${VALIDATION_ARGS[@]}"
