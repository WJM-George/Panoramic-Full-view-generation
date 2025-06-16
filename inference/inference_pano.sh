#!/bin/bash
# Script to run inference with CogVideoX for true V2V (requires retraining with base model)
# View of a panoramic scene from 0-90 degrees
#     --num_frames 49 \
python /home/kmwuab/CogVideo/inference/cli_demo.py \
    --prompt "View of a panoramic scene from 0-90 degrees base on the input video" \
    --model_path THUDM/CogVideoX-5B \
    --lora_path /home/kmwuab/CogVideo/CogVideo_history_Checkpoint/checkpoint-930 \
    --generate_type "v2v" \
    --image_or_video_path /home/kmwuab/CogVideo/CogVideo_dataset/360x_dataset_HR/cogvideo_dataset/input_videos_mp4/06eeb914-a805-479e-be3f-8157c95190b5/Segment_0000/input_0-90.mp4 \
    --output_path /home/kmwuab/CogVideo/output_v2v.mp4 \
    --num_frames 49 \
    --width 720 \
    --height 480 \
    --fps 8