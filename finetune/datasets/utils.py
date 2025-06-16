import logging
from pathlib import Path
from typing import List, Tuple

import cv2
import torch
from torchvision.transforms.functional import resize


# Must import after torch because this can sometimes lead to a nasty segmentation fault, or stack smashing error
# Very few bug reports but it happens. Look in decord Github issues for more relevant information.
import decord  # isort:skip

decord.bridge.set_bridge("torch")


##########  loaders  ##########

''''''
def load_prompts(prompt_path: Path) -> List[str]:
    with open(prompt_path, "r", encoding="utf-8") as file:
        return [line.strip() for line in file.readlines() if len(line.strip()) > 0]


def load_videos(video_path: Path) -> List[Path]:
    with open(video_path, "r", encoding="utf-8") as file:
        return [
            video_path.parent / line.strip() for line in file.readlines() if len(line.strip()) > 0
        ]


def load_images(image_path: Path) -> List[Path]:
    with open(image_path, "r", encoding="utf-8") as file:
        return [
            image_path.parent / line.strip() for line in file.readlines() if len(line.strip()) > 0
        ]


def load_images_from_videos(videos_path: List[Path]) -> List[Path]:
    first_frames_dir = videos_path[0].parent.parent / "first_frames"
    first_frames_dir.mkdir(exist_ok=True)

    first_frame_paths = []
    for video_path in videos_path:
        frame_path = first_frames_dir / f"{video_path.stem}.png"
        if frame_path.exists():
            first_frame_paths.append(frame_path)
            continue

        # Open video
        cap = cv2.VideoCapture(str(video_path))

        # Read first frame
        ret, frame = cap.read()
        if not ret:
            raise RuntimeError(f"Failed to read video: {video_path}")

        # Save frame as PNG with same name as video
        cv2.imwrite(str(frame_path), frame)
        logging.info(f"Saved first frame to {frame_path}")

        # Release video capture
        cap.release()

        first_frame_paths.append(frame_path)

    return first_frame_paths


##########  preprocessors  ##########


def preprocess_image_with_resize(
    image_path: Path | str,
    height: int,
    width: int,
) -> torch.Tensor:
    """
    Loads and resizes a single image.

    Args:
        image_path: Path to the image file.
        height: Target height for resizing.
        width: Target width for resizing.

    Returns:
        torch.Tensor: Image tensor with shape [C, H, W] where:
            C = number of channels (3 for RGB)
            H = height
            W = width
    """
    if isinstance(image_path, str):
        image_path = Path(image_path)
    image = cv2.imread(image_path.as_posix())
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (width, height))
    image = torch.from_numpy(image).float()
    image = image.permute(2, 0, 1).contiguous()
    return image

'''
## original one: 
def preprocess_video_with_resize(
    video_path: Path | str,
    max_num_frames: int,
    height: int,
    width: int,
) -> torch.Tensor:
    """
    Loads and resizes a single video.

    The function processes the video through these steps:
      1. If video frame count > max_num_frames, downsample frames evenly
      2. If video dimensions don't match (height, width), resize frames

    Args:
        video_path: Path to the video file.
        max_num_frames: Maximum number of frames to keep.
        height: Target height for resizing.
        width: Target width for resizing.

    Returns:
        A torch.Tensor with shape [F, C, H, W] where:
          F = number of frames
          C = number of channels (3 for RGB)
          H = height
          W = width
    """
    if isinstance(video_path, str):
        video_path = Path(video_path)
    video_reader = decord.VideoReader(uri=video_path.as_posix(), width=width, height=height)
    video_num_frames = len(video_reader)
    if video_num_frames < max_num_frames:
        # Get all frames first
        frames = video_reader.get_batch(list(range(video_num_frames)))
        # Repeat the last frame until we reach max_num_frames
        last_frame = frames[-1:]
        num_repeats = max_num_frames - video_num_frames
        repeated_frames = last_frame.repeat(num_repeats, 1, 1, 1)
        frames = torch.cat([frames, repeated_frames], dim=0)
        return frames.float().permute(0, 3, 1, 2).contiguous()
    else:
        indices = list(range(0, video_num_frames, video_num_frames // max_num_frames))
        frames = video_reader.get_batch(indices)
        frames = frames[:max_num_frames].float()
        frames = frames.permute(0, 3, 1, 2).contiguous()
        return frames

# changes here
def preprocess_video_with_buckets(
    input_video_path: Path,
    gt_video_path: Path,
    resolution_buckets: List[Tuple[int, int, int]],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Preprocesses a pair of videos (input and ground truth) using the same bucket,
    determined by the ground truth video.

    Args:
        input_video_path: Path to the input video.
        gt_video_path: Path to the ground truth video.
        resolution_buckets: List of (num_frames, height, width) buckets.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: (input_video, gt_video) with matching shapes [F, C, H, W].
        torch.Tensor: Video tensor with shape [F, C, H, W] where:
            F = number of frames
            C = number of channels (3 for RGB)
            H = height
            W = width

    The function processes the video through these steps:
        1. Finds nearest frame bucket <= video frame count
        2. Downsamples frames evenly to match bucket size
        3. Finds nearest resolution bucket based on dimensions
        4. Resizes frames to match bucket resolution
    """
    # Load ground truth video to determine bucket
    gt_reader = decord.VideoReader(uri=gt_video_path.as_posix())
    gt_num_frames = len(gt_reader)
    valid_buckets = [b for b in resolution_buckets if b[0] <= gt_num_frames]
    if not valid_buckets:
        raise ValueError(f"Ground truth video {gt_video_path} has too few frames for buckets {resolution_buckets}")
    
    nearest_frame_bucket = min(valid_buckets, key=lambda b: gt_num_frames - b[0])[0]
    frame_indices = list(range(0, gt_num_frames, gt_num_frames // nearest_frame_bucket))
    gt_frames = gt_reader.get_batch(frame_indices)[:nearest_frame_bucket].float()
    gt_frames = gt_frames.permute(0, 3, 1, 2).contiguous()
    
    nearest_res = min(
        resolution_buckets,
        key=lambda x: abs(x[1] - gt_frames.shape[2]) + abs(x[2] - gt_frames.shape[3])
    )
    target_height, target_width = nearest_res[1], nearest_res[2]
    gt_frames = torch.stack([resize(f, (target_height, target_width)) for f in gt_frames], dim=0)
    
    # Process input video with the same parameters
    input_reader = decord.VideoReader(uri=input_video_path.as_posix())
    input_num_frames = len(input_reader)
    frame_indices = list(range(0, input_num_frames, input_num_frames // nearest_frame_bucket))
    input_frames = input_reader.get_batch(frame_indices)[:nearest_frame_bucket].float()
    input_frames = input_frames.permute(0, 3, 1, 2).contiguous()
    input_frames = torch.stack([resize(f, (target_height, target_width)) for f in input_frames], dim=0)
    
    return input_frames, gt_frames
'''


## new one: 
def preprocess_video_with_resize(
    video_path: Path | str,
    max_num_frames: int,
    height: int,
    width: int,
) -> torch.Tensor:
    """
    Loads and resizes a single video.

    The function processes the video through these steps:
      1. If video frame count > max_num_frames, downsample frames evenly
      2. If video frame count < max_num_frames, repeat the last frame
      3. Resize frames to target height and width using PyTorch

    Args:
        video_path: Path to the video file.
        max_num_frames: Maximum number of frames to keep.
        height: Target height for resizing.
        width: Target width for resizing.

    Returns:
        A torch.Tensor with shape [F, C, H, W] where:
          F = number of frames
          C = number of channels (3 for RGB)
          H = height
          W = width
    """
    if isinstance(video_path, str):
        video_path = Path(video_path)
    
    # Load video without resizing
    try:
        video_reader = decord.VideoReader(uri=video_path.as_posix())
    except Exception as e:
        logging.error(f"Failed to load video {video_path}: {e}")
        raise
    
    video_num_frames = len(video_reader)
    
    if video_num_frames < max_num_frames:
        # Get all frames and repeat the last frame
        frames = video_reader.get_batch(list(range(video_num_frames)))
        last_frame = frames[-1:]
        num_repeats = max_num_frames - video_num_frames
        repeated_frames = last_frame.repeat(num_repeats, 0)
        frames = torch.cat([frames, repeated_frames], dim=0)
    else:
        # Sample frames evenly
        indices = list(range(0, video_num_frames, video_num_frames // max_num_frames))
        frames = video_reader.get_batch(indices)[:max_num_frames]
    
    # Convert to float and permute to [F, C, H, W]
    frames = frames.float().permute(0, 3, 1, 2)
    
    # Resize frames using PyTorch
    frames = torch.stack([resize(f, (height, width)) for f in frames], dim=0)
    
    return frames


'''
def preprocess_video_with_resize(
    video_path: Path | str,
    max_num_frames: int,
    height: int,
    width: int,
) -> torch.Tensor:
    if isinstance(video_path, str):
        video_path = Path(video_path)
    
    cap = cv2.VideoCapture(video_path.as_posix())
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = torch.from_numpy(frame).float()
        frame = frame.permute(2, 0, 1)  # [C, H, W]
        frames.append(frame)
    
    cap.release()
    
    if not frames:
        raise RuntimeError(f"No frames found in video: {video_path}")
    
    frames = torch.stack(frames)  # [F, C, H, W]
    
    # Adjust frame count
    video_num_frames = len(frames)
    if video_num_frames < max_num_frames:
        last_frame = frames[-1:]
        num_repeats = max_num_frames - video_num_frames
        repeated_frames = last_frame.repeat(num_repeats, 1, 1, 1)
        frames = torch.cat([frames, repeated_frames], dim=0)
    else:
        indices = list(range(0, video_num_frames, video_num_frames // max_num_frames))
        frames = frames[indices][:max_num_frames]
    
    # Resize frames
    frames = torch.stack([resize(f, (height, width)) for f in frames], dim=0)
    
    return frames
'''


def preprocess_video_with_buckets(
    input_video_path: Path,
    gt_video_path: Path,
    resolution_buckets: List[Tuple[int, int, int]],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Preprocesses a pair of videos (input and ground truth) using the same bucket,
    determined by the ground truth video.

    Args:
        input_video_path: Path to the input video.
        gt_video_path: Path to the ground truth video.
        resolution_buckets: List of (num_frames, height, width) buckets.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: (input_video, gt_video) with matching shapes [F, C, H, W].
    """
    # Load ground truth video to determine bucket
    gt_reader = decord.VideoReader(uri=gt_video_path.as_posix())
    gt_num_frames = len(gt_reader)
    valid_buckets = [b for b in resolution_buckets if b[0] <= gt_num_frames]
    if not valid_buckets:
        raise ValueError(f"Ground truth video {gt_video_path} has too few frames for buckets {resolution_buckets}")
    
    nearest_frame_bucket = min(valid_buckets, key=lambda b: gt_num_frames - b[0])[0]
    frame_indices = list(range(0, gt_num_frames, gt_num_frames // nearest_frame_bucket))
    gt_frames = gt_reader.get_batch(frame_indices)[:nearest_frame_bucket].float()
    gt_frames = gt_frames.permute(0, 3, 1, 2)
    
    nearest_res = min(
        resolution_buckets,
        key=lambda x: abs(x[1] - gt_frames.shape[2]) + abs(x[2] - gt_frames.shape[3])
    )
    target_height, target_width = nearest_res[1], nearest_res[2]
    gt_frames = torch.stack([resize(f, (target_height, target_width)) for f in gt_frames], dim=0)
    
    # Process input video with the same parameters
    input_reader = decord.VideoReader(uri=input_video_path.as_posix())
    input_num_frames = len(input_reader)
    frame_indices = list(range(0, input_num_frames, input_num_frames // nearest_frame_bucket))
    input_frames = input_reader.get_batch(frame_indices)[:nearest_frame_bucket].float()
    input_frames = input_frames.permute(0, 3, 1, 2)
    input_frames = torch.stack([resize(f, (target_height, target_width)) for f in input_frames], dim=0)
    
    return input_frames, gt_frames
