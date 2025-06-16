#!/bin/bash
# Script to run inference with CogVideoX using checkpoint-100 for I2V by extracting the first frame

# Path to your input video
INPUT_VIDEO="/home/kmwuab/CogVideo/CogVideo_dataset/360x_dataset_HR/cogvideo_dataset/input_videos_mp4/06eeb914-a805-479e-be3f-8157c95190b5/Segment_0001/input_90-180.mp4"
# Path to save the extracted first frame
INPUT_FRAME="./input_frame_0001.jpg"

# Extract the first frame using ffmpeg
ffmpeg -i "$INPUT_VIDEO" -vframes 1 -q:v 2 "$INPUT_FRAME"

# Run inference with the extracted frame
python /home/kmwuab/CogVideo/inference/cli_demo.py \
    --prompt "View of a panoramic scene from 0-90 degrees base on the input video, do not contains people inside, only the natural scene just like the input video." \
    --model_path THUDM/CogVideoX-5B-I2V \
    --lora_path /home/kmwuab/CogVideo/CogVideo_history_Checkpoint/checkpoint-1340 \
    --generate_type "i2v" \
    --image_or_video_path "$INPUT_FRAME" \
    --output_path /home/kmwuab/CogVideo/output_007_i2v.mp4 \
    --num_frames 49 \
    --width 720 \
    --height 480 \
    --fps 8