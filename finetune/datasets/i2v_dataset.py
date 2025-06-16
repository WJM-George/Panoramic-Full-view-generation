import hashlib
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Tuple

import torch
from accelerate.logging import get_logger
from safetensors.torch import load_file, save_file
from torch.utils.data import Dataset
from torchvision import transforms
from typing_extensions import override

from finetune.constants import LOG_LEVEL, LOG_NAME
import pdb
from .utils import (
    load_images,
    load_images_from_videos,
    load_prompts,
    load_videos,
    preprocess_image_with_resize,
    preprocess_video_with_buckets,
    preprocess_video_with_resize,
)

#preprocess_video_pair_with_buckets,


if TYPE_CHECKING:
    from finetune.trainer import Trainer

# Must import after torch because this can sometimes lead to a nasty segmentation fault, or stack smashing error
# Very few bug reports but it happens. Look in decord Github issues for more relevant information.
import decord  # isort:skip

# Wholesome transformation from i2v to v2v dataset loading.

decord.bridge.set_bridge("torch")

logger = get_logger(LOG_NAME, LOG_LEVEL)


class BaseI2VDataset(Dataset):
    """
    Base dataset class for Video-to-Video (V2V) training.

    Loads prompts, input videos (panning views), and ground truth videos (full views) for V2V training.

    Args:
        data_root (str): Root directory containing dataset files.
        caption_column (str): Path to file with text prompts.
        input_video_column (str): Path to file with input video paths (panning views).
        gt_video_column (str): Path to file with ground truth video paths (full views).
        device (torch.device): Device for data loading.
        trainer: Trainer object providing encoding functions.
    """
    def __init__(
        self,
        data_root: str,
        caption_column: str,
        input_video_column: str,
        gt_video_column: str,
        device: torch.device,
        trainer=None,
        *args,
        **kwargs,
    ) -> None:
        super().__init__()
        data_root = Path(data_root)
        self.prompts = load_prompts(data_root / caption_column)
        self.input_videos = load_videos(data_root / input_video_column)
        self.gt_videos = load_videos(data_root / gt_video_column)
        self.trainer = trainer
        self.device = device
        self.encode_video = trainer.encode_video
        self.encode_text = trainer.encode_text

        ###pdb.set_trace()
        # Validate lengths
        # it is fine here: 
        if not (len(self.input_videos) == len(self.gt_videos) == len(self.prompts)):
            raise ValueError(
                f"Lengths must match: {len(self.prompts)=}, {len(self.input_videos)=}, {len(self.gt_videos)=}"
            )
        # Check file existence
        for video_list, name in [(self.input_videos, "input"), (self.gt_videos, "ground truth")]:
            if any(not path.is_file() for path in video_list):
                raise ValueError(f"Missing {name} video file: {next(path for path in video_list if not path.is_file())}")

    def __len__(self) -> int:
        return len(self.gt_videos)

    def __getitem__(self, index: int) -> dict:
        if isinstance(index, list):
            return index  # For bucketing compatibility

        prompt = self.prompts[index]
        input_video_path = self.input_videos[index]
        gt_video_path = self.gt_videos[index]
        train_resolution_str = "x".join(str(x) for x in self.trainer.args.train_resolution)

        # Cache directories
        cache_dir = self.trainer.args.data_root / "cache"
        gt_video_latent_dir = cache_dir / "gt_video_latent" / self.trainer.args.model_name / train_resolution_str
        prompt_embeddings_dir = cache_dir / "prompt_embeddings"
        gt_video_latent_dir.mkdir(parents=True, exist_ok=True)
        prompt_embeddings_dir.mkdir(parents=True, exist_ok=True)

        # Prompt embedding
        prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()
        prompt_embedding_path = prompt_embeddings_dir / (prompt_hash + ".safetensors")
        if prompt_embedding_path.exists():
            prompt_embedding = load_file(prompt_embedding_path)["prompt_embedding"]
        else:
            prompt_embedding = self.encode_text(prompt).to("cpu")[0]  # [seq_len, hidden_size]
            save_file({"prompt_embedding": prompt_embedding}, prompt_embedding_path)

        # Ground truth video latent
        encoded_gt_video_path = gt_video_latent_dir / (gt_video_path.stem + ".safetensors")
        if encoded_gt_video_path.exists():
            encoded_gt_video = load_file(encoded_gt_video_path)["encoded_video"]
            # Load input video separately since ground truth is cached
            input_frames, _ = self.preprocess(input_video_path, None)
            input_frames = self.video_transform(input_frames)
            input_frames = input_frames.permute(1, 0, 2, 3)  # [C, F_in, H, W]
        else:
            # Process both videos together
            input_frames, gt_frames = self.preprocess(input_video_path, gt_video_path)
            gt_frames = gt_frames.to(self.device)
            input_frames = input_frames.to(self.device)
            gt_frames = self.video_transform(gt_frames)  # [F, C, H, W]
            input_frames = self.video_transform(input_frames)  # [F_in, C, H, W]
            gt_frames = gt_frames.unsqueeze(0).permute(0, 2, 1, 3, 4)  # [1, C, F, H, W]
            encoded_gt_video = self.encode_video(gt_frames)[0].to("cpu")  # [C, F, H_latent, W_latent]
            save_file({"encoded_video": encoded_gt_video}, encoded_gt_video_path)
            input_frames = input_frames.permute(1, 0, 2, 3)  # [C, F_in, H, W]

        return {
            "input_video": input_frames,  # [C, F_in, H, W]
            "prompt_embedding": prompt_embedding,  # [seq_len, hidden_size]
            "encoded_video": encoded_gt_video,  # [C, F, H_latent, W_latent]
            "video_metadata": {
                "num_frames": encoded_gt_video.shape[1],
                "height": encoded_gt_video.shape[2],
                "width": encoded_gt_video.shape[3],
            },
        }
    
    def preprocess(
        self, input_video_path: Path | None, gt_video_path: Path | None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Loads and preprocesses the input video and ground truth video.
        If either path is None, no preprocessing is done for that input.

        Args:
            input_video_path: Path to the input video file (panning view)
            gt_video_path: Path to the ground truth video file (full view)

        Returns:
            A tuple containing:
                - input_video (torch.Tensor): Shape [F, C, H, W], the input video frames
                - gt_video (torch.Tensor): Shape [F, C, H, W], the ground truth video frames
        """
        raise NotImplementedError("Subclass must implement this method")
        
    def video_transform(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Applies transformations to a video.

        Args:
            frames (torch.Tensor): A 4D tensor representing a video
                with shape [F, C, H, W] where:
                - F is number of frames
                - C is number of channels (3 for RGB)
                - H is height
                - W is width

        Returns:
            torch.Tensor: The transformed video tensor
        """
        raise NotImplementedError("Subclass must implement this method")

    def image_transform(self, image: torch.Tensor) -> torch.Tensor:
        """
        Applies transformations to an image.

        Args:
            image (torch.Tensor): A 3D tensor representing an image
                with shape [C, H, W] where:
                - C is number of channels (3 for RGB)
                - H is height
                - W is width

        Returns:
            torch.Tensor: The transformed image tensor
        """
        raise NotImplementedError("Subclass must implement this method")


class I2VDatasetWithResize(BaseI2VDataset):
    """
    A dataset class for image-to-video generation that resizes inputs to fixed dimensions.

    This class preprocesses videos and images by resizing them to specified dimensions:
    - Videos are resized to max_num_frames x height x width
    - Images are resized to height x width

    Args:
        max_num_frames (int): Maximum number of frames to extract from videos
        height (int): Target height for resizing videos and images
        width (int): Target width for resizing videos and images
    """

    def __init__(self, max_num_frames: int, height: int, width: int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.max_num_frames = max_num_frames
        self.height = height
        self.width = width

        self.__frame_transforms = transforms.Compose(
            [transforms.Lambda(lambda x: x / 255.0 * 2.0 - 1.0)]
        )
        self.__image_transforms = self.__frame_transforms

    @override
    def preprocess(
        self, input_video_path: Path | None, gt_video_path: Path | None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        input_video = None
        gt_video = None
        if input_video_path is not None:
            input_video = preprocess_video_with_resize(
                input_video_path, self.max_num_frames, self.height, self.width
            )
        if gt_video_path is not None:
            gt_video = preprocess_video_with_resize(
                gt_video_path, self.max_num_frames, self.height, self.width
            )
        # If only one video is provided, ensure the other matches its shape
        if input_video is None and gt_video is not None:
            input_video = torch.zeros_like(gt_video)
        elif gt_video is None and input_video is not None:
            gt_video = torch.zeros_like(input_video)
        return input_video, gt_video

    @override
    def video_transform(self, frames: torch.Tensor) -> torch.Tensor:
        return torch.stack([self.__frame_transforms(f) for f in frames], dim=0)

    @override
    def image_transform(self, image: torch.Tensor) -> torch.Tensor:
        return self.__image_transforms(image)


class I2VDatasetWithBuckets(BaseI2VDataset):
    def __init__(
        self,
        video_resolution_buckets: List[Tuple[int, int, int]],
        vae_temporal_compression_ratio: int,
        vae_height_compression_ratio: int,
        vae_width_compression_ratio: int,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.video_resolution_buckets = [
            (
                int(b[0] / vae_temporal_compression_ratio),
                int(b[1] / vae_height_compression_ratio),
                int(b[2] / vae_width_compression_ratio),
            )
            for b in video_resolution_buckets
        ]
        self.__frame_transforms = transforms.Compose(
            [transforms.Lambda(lambda x: x / 255.0 * 2.0 - 1.0)]
        )
        self.__image_transforms = self.__frame_transforms

    @override
    def preprocess(
        self, input_video_path: Path | None, gt_video_path: Path | None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        input_video = None
        gt_video = None
        if input_video_path is not None and gt_video_path is not None:
            input_video, gt_video = preprocess_video_pair_with_buckets(
                input_video_path, gt_video_path, self.video_resolution_buckets
            )
        elif input_video_path is not None:
            input_video = preprocess_video_with_buckets(input_video_path, self.video_resolution_buckets)
            gt_video = torch.zeros_like(input_video)
        elif gt_video_path is not None:
            gt_video = preprocess_video_with_buckets(gt_video_path, self.video_resolution_buckets)
            input_video = torch.zeros_like(gt_video)
        return input_video, gt_video

    @override
    def video_transform(self, frames: torch.Tensor) -> torch.Tensor:
        return torch.stack([self.__frame_transforms(f) for f in frames], dim=0)

    @override
    def image_transform(self, image: torch.Tensor) -> torch.Tensor:
        return self.__image_transforms(image)
