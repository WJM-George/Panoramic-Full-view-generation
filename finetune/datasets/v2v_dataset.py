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

from .utils import (
    load_images,
    load_images_from_videos,
    load_prompts,
    load_videos,
    preprocess_image_with_resize,
    preprocess_video_with_buckets,
    preprocess_video_with_resize,
)

if TYPE_CHECKING:
    from finetune.trainer import Trainer

import decord  # isort:skip
decord.bridge.set_bridge("torch")

logger = get_logger(LOG_NAME, LOG_LEVEL)

class BaseI2VDataset(Dataset):
    """
    Base dataset class for Image-to-Video (I2V) training.

    This dataset loads prompts, videos and corresponding conditioning images for I2V training.

    Args:
        data_root (str): Root directory containing the dataset files
        caption_column (str): Path to file containing text prompts/captions
        video_column (str): Path to file containing video paths
        image_column (str): Path to file containing image paths
        device (torch.device): Device to load the data on
        trainer (Trainer): Trainer instance providing encoding functions
    """
    def __init__(
        self,
        data_root: str,
        caption_column: str,
        video_column: str,
        image_column: str | None,
        device: torch.device,
        trainer: "Trainer" = None,
        *args,
        **kwargs,
    ) -> None:
        super().__init__()
        data_root = Path(data_root)
        self.prompts = load_prompts(data_root / caption_column)
        self.videos = load_videos(data_root / video_column)
        if image_column is not None:
            self.images = load_images(data_root / image_column)
        else:
            self.images = load_images_from_videos(self.videos)
        self.trainer = trainer
        self.device = device
        self.encode_video = trainer.encode_video
        self.encode_text = trainer.encode_text

        if not (len(self.videos) == len(self.prompts) == len(self.images)):
            raise ValueError(
                f"Expected length of prompts, videos and images to be the same but found {len(self.prompts)=}, {len(self.videos)=} and {len(self.images)=}."
            )

        if any(not path.is_file() for path in self.videos):
            raise ValueError(f"Missing video file: {next(path for path in self.videos if not path.is_file())}")

        if any(not path.is_file() for path in self.images):
            raise ValueError(f"Missing image file: {next(path for path in self.images if not path.is_file())}")

    def __len__(self) -> int:
        return len(self.videos)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        if isinstance(index, list):
            return index

        prompt = self.prompts[index]
        video = self.videos[index]
        image = self.images[index]
        train_resolution_str = "x".join(str(x) for x in self.trainer.args.train_resolution)

        cache_dir = self.trainer.args.data_root / "cache"
        video_latent_dir = cache_dir / "video_latent" / self.trainer.args.model_name / train_resolution_str
        prompt_embeddings_dir = cache_dir / "prompt_embeddings"
        video_latent_dir.mkdir(parents=True, exist_ok=True)
        prompt_embeddings_dir.mkdir(parents=True, exist_ok=True)

        prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()
        prompt_embedding_path = prompt_embeddings_dir / f"{prompt_hash}.safetensors"
        encoded_video_path = video_latent_dir / f"{video.stem}.safetensors"

        if prompt_embedding_path.exists():
            prompt_embedding = load_file(prompt_embedding_path)["prompt_embedding"]
        else:
            prompt_embedding = self.encode_text(prompt).to("cpu")[0]
            save_file({"prompt_embedding": prompt_embedding}, prompt_embedding_path)

        if encoded_video_path.exists():
            encoded_video = load_file(encoded_video_path)["encoded_video"]
            _, image = self.preprocess(None, self.images[index])
            image = self.image_transform(image)
        else:
            frames, image = self.preprocess(video, image)
            frames = frames.to(self.device)
            image = self.image_transform(image.to(self.device))
            frames = self.video_transform(frames)
            frames = frames.unsqueeze(0).permute(0, 2, 1, 3, 4).contiguous()
            encoded_video = self.encode_video(frames)[0].to("cpu")
            image = image.to("cpu")
            save_file({"encoded_video": encoded_video}, encoded_video_path)

        return {
            "image": image,
            "prompt_embedding": prompt_embedding,
            "encoded_video": encoded_video,
            "video_metadata": {
                "num_frames": encoded_video.shape[1],
                "height": encoded_video.shape[2],
                "width": encoded_video.shape[3],
            },
        }

    def preprocess(self, video_path: Path | None, image_path: Path | None) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError("Subclass must implement this method")

    def video_transform(self, frames: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Subclass must implement this method")

    def image_transform(self, image: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Subclass must implement this method")

class I2VDatasetWithResize(BaseI2VDataset):
    """
    A dataset class for image-to-video generation that resizes inputs to fixed dimensions.

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
        self.__frame_transforms = transforms.Compose([transforms.Lambda(lambda x: x / 255.0 * 2.0 - 1.0)])
        self.__image_transforms = self.__frame_transforms

    @override
    def preprocess(self, video_path: Path | None, image_path: Path | None) -> Tuple[torch.Tensor, torch.Tensor]:
        video = preprocess_video_with_resize(video_path, self.max_num_frames, self.height, self.width) if video_path else None
        image = preprocess_image_with_resize(image_path, self.height, self.width) if image_path else None
        return video, image

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
        self.__frame_transforms = transforms.Compose([transforms.Lambda(lambda x: x / 255.0 * 2.0 - 1.0)])
        self.__image_transforms = self.__frame_transforms

    @override
    def preprocess(self, video_path: Path, image_path: Path) -> Tuple[torch.Tensor, torch.Tensor]:
        video = preprocess_video_with_buckets(video_path, self.video_resolution_buckets)
        image = preprocess_image_with_resize(image_path, video.shape[2], video.shape[3])
        return video, image

    @override
    def video_transform(self, frames: torch.Tensor) -> torch.Tensor:
        return torch.stack([self.__frame_transforms(f) for f in frames], dim=0)

    @override
    def image_transform(self, image: torch.Tensor) -> torch.Tensor:
        return self.__image_transforms(image)

class V2VDatasetWithResize(Dataset):
    """
    A dataset class for video-to-video generation that resizes inputs to fixed dimensions.
    
    This class preprocesses an input video (partial view with sliding window) and a target video (full view) to specified dimensions:
    - Input videos are resized to condition_frames x height x width
    - Target videos are resized to max_num_frames x height x width
    
    Args:
        data_root (str): Root directory containing the dataset files
        caption_column (str): Path to file containing text prompts/captions
        input_video_column (str): Path to file containing input video paths (partial view)
        target_video_column (str): Path to file containing target video paths (full view)
        device (torch.device): Device to load the data on
        trainer (Trainer): Trainer instance providing encoding functions
        max_num_frames (int): Maximum number of frames for target videos
        condition_frames (int): Number of frames for input videos
        height (int): Target height for resizing videos
        width (int): Target width for resizing videos
    """
    def __init__(
        self,
        data_root: str,
        caption_column: str,
        input_video_column: str,
        target_video_column: str,
        device: torch.device,
        trainer: "Trainer",
        max_num_frames: int,
        condition_frames: int,
        height: int,
        width: int,
        *args,
        **kwargs,
    ) -> None:
        super().__init__()
        data_root = Path(data_root)
        self.prompts = load_prompts(data_root / caption_column)
        self.input_videos = load_videos(data_root / input_video_column)  # Partial view videos with sliding window
        self.target_videos = load_videos(data_root / target_video_column)  # Full view videos
        self.trainer = trainer
        self.device = device
        self.encode_video = trainer.encode_video
        self.encode_text = trainer.encode_text
        self.max_num_frames = max_num_frames
        self.condition_frames = condition_frames
        self.height = height
        self.width = width
        self.__frame_transforms = transforms.Compose(
            [transforms.Lambda(lambda x: x / 255.0 * 2.0 - 1.0)]
        )

        if not (len(self.prompts) == len(self.input_videos) == len(self.target_videos)):
            raise ValueError(
                f"Lengths mismatch: prompts={len(self.prompts)}, input_videos={len(self.input_videos)}, target_videos={len(self.target_videos)}"
            )

        for video_list, name in [(self.input_videos, "input_videos"), (self.target_videos, "target_videos")]:
            if any(not path.is_file() for path in video_list):
                raise ValueError(f"Missing {name} file: {next(path for path in video_list if not path.is_file())}")

    def __len__(self) -> int:
        return len(self.prompts)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        prompt = self.prompts[index]
        input_video_path = self.input_videos[index]
        target_video_path = self.target_videos[index]
        train_resolution_str = "x".join(str(x) for x in [self.condition_frames, self.height, self.width])

        # Cache directories
        cache_dir = self.trainer.args.data_root / "cache"
        input_video_latent_dir = cache_dir / "input_video_latent" / self.trainer.args.model_name / train_resolution_str
        target_video_latent_dir = cache_dir / "target_video_latent" / self.trainer.args.model_name / train_resolution_str
        prompt_embeddings_dir = cache_dir / "prompt_embeddings"
        for d in [input_video_latent_dir, target_video_latent_dir, prompt_embeddings_dir]:
            d.mkdir(parents=True, exist_ok=True)

        # File paths
        prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()
        prompt_embedding_path = prompt_embeddings_dir / f"{prompt_hash}.safetensors"
        encoded_input_video_path = input_video_latent_dir / f"{input_video_path.stem}.safetensors"
        encoded_target_video_path = target_video_latent_dir / f"{target_video_path.stem}.safetensors"

        # Load or compute prompt embedding
        if prompt_embedding_path.exists():
            prompt_embedding = load_file(prompt_embedding_path)["prompt_embedding"]
        else:
            prompt_embedding = self.encode_text(prompt).to("cpu")[0]  # [1, seq_len, dim] -> [seq_len, dim]
            save_file({"prompt_embedding": prompt_embedding}, prompt_embedding_path)

        # Load or compute input video latent (partial view with sliding window)
        if encoded_input_video_path.exists():
            encoded_input_video = load_file(encoded_input_video_path)["encoded_video"]
        else:
            input_frames = preprocess_video_with_resize(input_video_path, self.condition_frames, self.height, self.width)
            input_frames = self.video_transform(input_frames.to(self.device))
            input_frames = input_frames.unsqueeze(0).permute(0, 2, 1, 3, 4).contiguous()  # [1, C, F, H, W]
            encoded_input_video = self.encode_video(input_frames)[0].to("cpu")  # [C, F, H, W]
            save_file({"encoded_video": encoded_input_video}, encoded_input_video_path)

        # Load or compute target video latent (full view)
        if encoded_target_video_path.exists():
            encoded_target_video = load_file(encoded_target_video_path)["encoded_video"]
        else:
            target_frames = preprocess_video_with_resize(target_video_path, self.max_num_frames, self.height, self.width)
            target_frames = self.video_transform(target_frames.to(self.device))
            target_frames = target_frames.unsqueeze(0).permute(0, 2, 1, 3, 4).contiguous()  # [1, C, F, H, W]
            encoded_target_video = self.encode_video(target_frames)[0].to("cpu")  # [C, F, H, W]
            save_file({"encoded_video": encoded_target_video}, encoded_target_video_path)

        return {
            "encoded_input_video": encoded_input_video,
            "prompt_embedding": prompt_embedding,
            "encoded_target_video": encoded_target_video,
            "video_metadata": {
                "num_frames": encoded_target_video.shape[1],
                "height": encoded_target_video.shape[2],
                "width": encoded_target_video.shape[3],
            },
        }

    def video_transform(self, frames: torch.Tensor) -> torch.Tensor:
        return torch.stack([self.__frame_transforms(f) for f in frames], dim=0)