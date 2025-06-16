import argparse
import logging
import math
import os
import random
import shutil
from datetime import timedelta
from pathlib import Path
from typing import List, Optional, Tuple, Union

import torch
import transformers
from accelerate import Accelerator, DistributedType
from accelerate.logging import get_logger
from accelerate.utils import DistributedDataParallelKwargs, InitProcessGroupKwargs, ProjectConfiguration, set_seed
from huggingface_hub import create_repo, upload_folder
from peft import LoraConfig, get_peft_model_state_dict, set_peft_model_state_dict
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import AutoTokenizer, T5EncoderModel, T5Tokenizer

import diffusers
from diffusers import (
    AutoencoderKLCogVideoX,
    CogVideoXDPMScheduler,
    CogVideoXTransformer3DModel,
)
from diffusers.models.embeddings import get_3d_rotary_pos_embed
from diffusers.optimization import get_scheduler
from diffusers.training_utils import cast_training_params, free_memory
from diffusers.utils import (
    check_min_version,
    convert_unet_state_dict_to_peft,
    export_to_video,
    is_wandb_available,
)
from diffusers.utils.hub_utils import load_or_create_model_card, populate_model_card
from diffusers.utils.torch_utils import is_compiled_module

if is_wandb_available():
    import wandb

check_min_version("0.34.0.dev0")

logger = get_logger(__name__)

def get_args():
    parser = argparse.ArgumentParser(description="Training script for CogVideoX V2V LoRA.")
    # Model information (keep as is)
    parser.add_argument("--pretrained_model_name_or_path", type=str, default=None, required=True)
    parser.add_argument("--revision", type=str, default=None)
    parser.add_argument("--variant", type=str, default=None)
    parser.add_argument("--cache_dir", type=str, default=None)

    # Dataset information (modify for V2V)
    parser.add_argument("--dataset_name", type=str, default=None)
    parser.add_argument("--dataset_config_name", type=str, default=None)
    parser.add_argument("--instance_data_root", type=str, default=None)
    parser.add_argument("--input_video_column", type=str, default="input_video", help="Column/file for input videos")
    parser.add_argument("--target_video_column", type=str, default="target_video", help="Column/file for target videos")
    parser.add_argument("--caption_column", type=str, default="text")
    parser.add_argument("--id_token", type=str, default=None)
    parser.add_argument("--dataloader_num_workers", type=int, default=0)

    # Validation (modify for V2V)
    parser.add_argument("--validation_prompt", type=str, default=None)
    parser.add_argument("--validation_input_videos", type=str, default=None, help="Paths to input videos for validation")
    parser.add_argument("--validation_prompt_separator", type=str, default=":::")
    parser.add_argument("--num_validation_videos", type=int, default=1)
    parser.add_argument("--validation_epochs", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=6)
    parser.add_argument("--use_dynamic_cfg", action="store_true", default=False)

    # Training information (keep as is)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--rank", type=int, default=128)
    parser.add_argument("--lora_alpha", type=float, default=128)
    parser.add_argument("--mixed_precision", type=str, default=None, choices=["no", "fp16", "bf16"])
    parser.add_argument("--output_dir", type=str, default="cogvideox-v2v-lora")
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=720)
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--max_num_frames", type=int, default=49)
    parser.add_argument("--skip_frames_start", type=int, default=0)
    parser.add_argument("--skip_frames_end", type=int, default=0)
    parser.add_argument("--random_flip", action="store_true")
    parser.add_argument("--train_batch_size", type=int, default=4)
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--max_train_steps", type=int, default=None)
    parser.add_argument("--checkpointing_steps", type=int, default=500)
    parser.add_argument("--checkpoints_total_limit", type=int, default=None)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--scale_lr", action="store_true", default=False)
    parser.add_argument("--lr_scheduler", type=str, default="constant")
    parser.add_argument("--lr_warmup_steps", type=int, default=500)
    parser.add_argument("--lr_num_cycles", type=int, default=1)
    parser.add_argument("--lr_power", type=float, default=1.0)
    parser.add_argument("--enable_slicing", action="store_true", default=False)
    parser.add_argument("--enable_tiling", action="store_true", default=False)
    parser.add_argument("--noised_video_dropout", type=float, default=0.05, help="Input video dropout probability")

    # Optimizer (keep as is)
    parser.add_argument("--optimizer", type=lambda s: s.lower(), default="adam", choices=["adam", "adamw", "prodigy"])
    parser.add_argument("--use_8bit_adam", action="store_true")
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.95)
    parser.add_argument("--prodigy_beta3", type=float, default=None)
    parser.add_argument("--prodigy_decouple", action="store_true")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-04)
    parser.add_argument("--adam_epsilon", type=float, default=1e-08)
    parser.add_argument("--max_grad_norm", default=1.0, type=float)
    parser.add_argument("--prodigy_use_bias_correction", action="store_true")
    parser.add_argument("--prodigy_safeguard_warmup", action="store_true")

    # Other information (keep as is)
    parser.add_argument("--tracker_name", type=str, default=None)
    parser.add_argument("--push_to_hub", action="store_true")
    parser.add_argument("--hub_token", type=str, default=None)
    parser.add_argument("--hub_model_id", type=str, default=None)
    parser.add_argument("--logging_dir", type=str, default="logs")
    parser.add_argument("--allow_tf32", action="store_true")
    parser.add_argument("--report_to", type=str, default=None)
    parser.add_argument("--nccl_timeout", type=int, default=600)

    return parser.parse_args()

# new setting?
class VideoDataset(Dataset):
    def __init__(
        self,
        instance_data_root: Optional[str] = None,
        dataset_name: Optional[str] = None,
        dataset_config_name: Optional[str] = None,
        caption_column: str = "text",
        input_video_column: str = "input_video",
        target_video_column: str = "target_video",
        height: int = 480,
        width: int = 720,
        fps: int = 8,
        max_num_frames: int = 49,
        skip_frames_start: int = 0,
        skip_frames_end: int = 0,
        cache_dir: Optional[str] = None,
        id_token: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.instance_data_root = Path(instance_data_root) if instance_data_root is not None else None
        self.dataset_name = dataset_name
        self.dataset_config_name = dataset_config_name
        self.caption_column = caption_column
        self.input_video_column = input_video_column
        self.target_video_column = target_video_column
        self.height = height
        self.width = width
        self.fps = fps
        self.max_num_frames = max_num_frames
        self.skip_frames_start = skip_frames_start
        self.skip_frames_end = skip_frames_end
        self.cache_dir = cache_dir
        self.id_token = id_token or ""

        if dataset_name is not None:
            self.instance_prompts, self.instance_input_videos, self.instance_target_videos = self._load_dataset_from_hub()
        else:
            self.instance_prompts, self.instance_input_videos, self.instance_target_videos = self._load_dataset_from_local_path()

        self.instance_prompts = [self.id_token + prompt for prompt in self.instance_prompts]
        self.num_instance_videos = len(self.instance_input_videos)
        if not (self.num_instance_videos == len(self.instance_prompts) == len(self.instance_target_videos)):
            raise ValueError("Mismatch in lengths of prompts, input videos, and target videos.")

        self.instance_input_videos, self.instance_target_videos = self._preprocess_data()

    def __len__(self):
        return self.num_instance_videos

    def __getitem__(self, index):
        return {
            "instance_prompt": self.instance_prompts[index],
            "instance_input_video": self.instance_input_videos[index],
            "instance_target_video": self.instance_target_videos[index],
        }

    def _load_dataset_from_hub(self):
        from datasets import load_dataset
        dataset = load_dataset(self.dataset_name, self.dataset_config_name, cache_dir=self.cache_dir)
        column_names = dataset["train"].column_names

        for col, name in [(self.input_video_column, "input_video_column"), (self.target_video_column, "target_video_column"), (self.caption_column, "caption_column")]:
            if col not in column_names:
                raise ValueError(f"`{name}` '{col}' not found in dataset columns: {', '.join(column_names)}")

        instance_prompts = dataset["train"][self.caption_column]
        instance_input_videos = [Path(self.instance_data_root, filepath) for filepath in dataset["train"][self.input_video_column]]
        instance_target_videos = [Path(self.instance_data_root, filepath) for filepath in dataset["train"][self.target_video_column]]
        return instance_prompts, instance_input_videos, instance_target_videos

    def _load_dataset_from_local_path(self):
        if not self.instance_data_root.exists():
            raise ValueError("Instance data root folder does not exist")

        prompt_path = self.instance_data_root.joinpath(self.caption_column)
        input_video_path = self.instance_data_root.joinpath(self.input_video_column)
        target_video_path = self.instance_data_root.joinpath(self.target_video_column)

        for path, name in [(prompt_path, "caption_column"), (input_video_path, "input_video_column"), (target_video_path, "target_video_column")]:
            if not path.exists() or not path.is_file():
                raise ValueError(f"Expected `{name}` to be a file in `--instance_data_root`.")

        with open(prompt_path, "r", encoding="utf-8") as file:
            instance_prompts = [line.strip() for line in file.readlines() if line.strip()]
        with open(input_video_path, "r", encoding="utf-8") as file:
            instance_input_videos = [self.instance_data_root.joinpath(line.strip()) for line in file.readlines() if line.strip()]
        with open(target_video_path, "r", encoding="utf-8") as file:
            instance_target_videos = [self.instance_data_root.joinpath(line.strip()) for line in file.readlines() if line.strip()]

        return instance_prompts, instance_input_videos, instance_target_videos

    def _preprocess_data(self):
        import decord
        decord.bridge.set_bridge("torch")

        input_videos, target_videos = [], []
        train_transforms = transforms.Compose([transforms.Lambda(lambda x: x / 255.0 * 2.0 - 1.0)])

        for input_path, target_path in zip(self.instance_input_videos, self.instance_target_videos):
            for path, video_list in [(input_path, input_videos), (target_path, target_videos)]:
                video_reader = decord.VideoReader(uri=path.as_posix(), width=self.width, height=self.height)
                video_num_frames = len(video_reader)

                start_frame = min(self.skip_frames_start, video_num_frames)
                end_frame = max(0, video_num_frames - self.skip_frames_end)
                if end_frame <= start_frame:
                    frames = video_reader.get_batch([start_frame])
                elif end_frame - start_frame <= self.max_num_frames:
                    frames = video_reader.get_batch(list(range(start_frame, end_frame)))
                else:
                    indices = list(range(start_frame, end_frame, (end_frame - start_frame) // self.max_num_frames))
                    frames = video_reader.get_batch(indices)

                frames = frames[:self.max_num_frames]
                selected_num_frames = frames.shape[0]
                remainder = (3 + (selected_num_frames % 4)) % 4
                if remainder != 0:
                    frames = frames[:-remainder]

                frames = frames.float()
                frames = torch.stack([train_transforms(frame) for frame in frames], dim=0)
                video_list.append(frames.permute(0, 3, 1, 2).contiguous())

        return input_videos, target_videos

def save_model_card(
    repo_id: str,
    videos=None,
    base_model: str = None,
    validation_prompt=None,
    repo_folder=None,
    fps=8,
):
    widget_dict = []
    if videos is not None:
        for i, video in enumerate(videos):
            video_path = f"final_video_{i}.mp4"
            export_to_video(video, os.path.join(repo_folder, video_path, fps=fps))
            widget_dict.append(
                {"text": validation_prompt if validation_prompt else " ", "output": {"url": video_path}},
            )

    model_description = f"""
# CogVideoX LoRA - {repo_id}

<Gallery />

## Model description

These are {repo_id} LoRA weights for {base_model}.

The weights were trained using the [CogVideoX Diffusers trainer](https://github.com/huggingface/diffusers/blob/main/examples/cogvideo/train_cogvideox_image_to_video_lora.py).

Was LoRA for the text encoder enabled? No.

## Download model

[Download the *.safetensors LoRA]({repo_id}/tree/main) in the Files & versions tab.

## Use it with the [🧨 diffusers library](https://github.com/huggingface/diffusers)

```py
import torch
from diffusers import CogVideoXImageToVideoPipeline
from diffusers.utils import load_image, export_to_video

pipe = CogVideoXImageToVideoPipeline.from_pretrained("THUDM/CogVideoX-5b", torch_dtype=torch.bfloat16).to("cuda")
pipe.load_lora_weights("{repo_id}", weight_name="pytorch_lora_weights.safetensors", adapter_name=["cogvideox-i2v-lora"])

# The LoRA adapter weights are determined by what was used for training.
# In this case, we assume `--lora_alpha` is 32 and `--rank` is 64.
# It can be made lower or higher from what was used in training to decrease or amplify the effect
# of the LoRA upto a tolerance, beyond which one might notice no effect at all or overflows.
pipe.set_adapters(["cogvideox-i2v-lora"], [32 / 64])

image = load_image("/path/to/image")
video = pipe(image=image, "{validation_prompt}", guidance_scale=6, use_dynamic_cfg=True).frames[0]
export_to_video(video, "output.mp4", fps=8)
```

For more details, including weighting, merging and fusing LoRAs, check the [documentation on loading LoRAs in diffusers](https://huggingface.co/docs/diffusers/main/en/using-diffusers/loading_adapters)

## License

Please adhere to the licensing terms as described [here](https://huggingface.co/THUDM/CogVideoX-5b-I2V/blob/main/LICENSE).
"""
    model_card = load_or_create_model_card(
        repo_id_or_path=repo_id,
        from_training=True,
        license="other",
        base_model=base_model,
        prompt=validation_prompt,
        model_description=model_description,
        widget=widget_dict,
    )
    tags = [
        "image-to-video",
        "diffusers-training",
        "diffusers",
        "lora",
        "cogvideox",
        "cogvideox-diffusers",
        "template:sd-lora",
    ]

    model_card = populate_model_card(model_card, tags=tags)
    model_card.save(os.path.join(repo_folder, "README.md"))


def log_validation(
    pipe,
    args,
    accelerator,
    pipeline_args,
    epoch,
    is_final_validation: bool = False,
):
    logger.info(
        f"Running validation... \n Generating {args.num_validation_videos} videos with prompt: {pipeline_args['prompt']}."
    )
    # We train on the simplified learning objective. If we were previously predicting a variance, we need the scheduler to ignore it
    scheduler_args = {}

    if "variance_type" in pipe.scheduler.config:
        variance_type = pipe.scheduler.config.variance_type

        if variance_type in ["learned", "learned_range"]:
            variance_type = "fixed_small"

        scheduler_args["variance_type"] = variance_type

    pipe.scheduler = CogVideoXDPMScheduler.from_config(pipe.scheduler.config, **scheduler_args)
    pipe = pipe.to(accelerator.device)
    # pipe.set_progress_bar_config(disable=True)

    # run inference
    generator = torch.Generator(device=accelerator.device).manual_seed(args.seed) if args.seed is not None else None

    videos = []
    for _ in range(args.num_validation_videos):
        video = pipe(**pipeline_args, generator=generator, output_type="np").frames[0]
        videos.append(video)

    for tracker in accelerator.trackers:
        phase_name = "test" if is_final_validation else "validation"
        if tracker.name == "wandb":
            video_filenames = []
            for i, video in enumerate(videos):
                prompt = (
                    pipeline_args["prompt"][:25]
                    .replace(" ", "_")
                    .replace(" ", "_")
                    .replace("'", "_")
                    .replace('"', "_")
                    .replace("/", "_")
                )
                filename = os.path.join(args.output_dir, f"{phase_name}_video_{i}_{prompt}.mp4")
                export_to_video(video, filename, fps=8)
                video_filenames.append(filename)

            tracker.log(
                {
                    phase_name: [
                        wandb.Video(filename, caption=f"{i}: {pipeline_args['prompt']}")
                        for i, filename in enumerate(video_filenames)
                    ]
                }
            )

    del pipe
    free_memory()

    return videos


def _get_t5_prompt_embeds(
    tokenizer: T5Tokenizer,
    text_encoder: T5EncoderModel,
    prompt: Union[str, List[str]],
    num_videos_per_prompt: int = 1,
    max_sequence_length: int = 226,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    text_input_ids=None,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)

    if tokenizer is not None:
        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
    else:
        if text_input_ids is None:
            raise ValueError("`text_input_ids` must be provided when the tokenizer is not specified.")

    prompt_embeds = text_encoder(text_input_ids.to(device))[0]
    prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

    # duplicate text embeddings for each generation per prompt, using mps friendly method
    _, seq_len, _ = prompt_embeds.shape
    prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(batch_size * num_videos_per_prompt, seq_len, -1)

    return prompt_embeds


def encode_prompt(
    tokenizer: T5Tokenizer,
    text_encoder: T5EncoderModel,
    prompt: Union[str, List[str]],
    num_videos_per_prompt: int = 1,
    max_sequence_length: int = 226,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    text_input_ids=None,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    prompt_embeds = _get_t5_prompt_embeds(
        tokenizer,
        text_encoder,
        prompt=prompt,
        num_videos_per_prompt=num_videos_per_prompt,
        max_sequence_length=max_sequence_length,
        device=device,
        dtype=dtype,
        text_input_ids=text_input_ids,
    )
    return prompt_embeds


def compute_prompt_embeddings(
    tokenizer, text_encoder, prompt, max_sequence_length, device, dtype, requires_grad: bool = False
):
    if requires_grad:
        prompt_embeds = encode_prompt(
            tokenizer,
            text_encoder,
            prompt,
            num_videos_per_prompt=1,
            max_sequence_length=max_sequence_length,
            device=device,
            dtype=dtype,
        )
    else:
        with torch.no_grad():
            prompt_embeds = encode_prompt(
                tokenizer,
                text_encoder,
                prompt,
                num_videos_per_prompt=1,
                max_sequence_length=max_sequence_length,
                device=device,
                dtype=dtype,
            )
    return prompt_embeds


def prepare_rotary_positional_embeddings(
    height: int,
    width: int,
    num_frames: int,
    vae_scale_factor_spatial: int = 8,
    patch_size: int = 2,
    attention_head_dim: int = 64,
    device: Optional[torch.device] = None,
    base_height: int = 480,
    base_width: int = 720,
) -> Tuple[torch.Tensor, torch.Tensor]:
    grid_height = height // (vae_scale_factor_spatial * patch_size)
    grid_width = width // (vae_scale_factor_spatial * patch_size)
    base_size_width = base_width // (vae_scale_factor_spatial * patch_size)
    base_size_height = base_height // (vae_scale_factor_spatial * patch_size)

    grid_crops_coords = get_resize_crop_region_for_grid((grid_height, grid_width), base_size_width, base_size_height)
    freqs_cos, freqs_sin = get_3d_rotary_pos_embed(
        embed_dim=attention_head_dim,
        crops_coords=grid_crops_coords,
        grid_size=(grid_height, grid_width),
        temporal_size=num_frames,
        device=device,
    )

    return freqs_cos, freqs_sin


def get_optimizer(args, params_to_optimize, use_deepspeed: bool = False):
    # Use DeepSpeed optimzer
    if use_deepspeed:
        from accelerate.utils import DummyOptim

        return DummyOptim(
            params_to_optimize,
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            eps=args.adam_epsilon,
            weight_decay=args.adam_weight_decay,
        )

    # Optimizer creation
    supported_optimizers = ["adam", "adamw", "prodigy"]
    if args.optimizer not in supported_optimizers:
        logger.warning(
            f"Unsupported choice of optimizer: {args.optimizer}. Supported optimizers include {supported_optimizers}. Defaulting to AdamW"
        )
        args.optimizer = "adamw"

    if args.use_8bit_adam and args.optimizer.lower() not in ["adam", "adamw"]:
        logger.warning(
            f"use_8bit_adam is ignored when optimizer is not set to 'Adam' or 'AdamW'. Optimizer was "
            f"set to {args.optimizer.lower()}"
        )

    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
            )

    if args.optimizer.lower() == "adamw":
        optimizer_class = bnb.optim.AdamW8bit if args.use_8bit_adam else torch.optim.AdamW

        optimizer = optimizer_class(
            params_to_optimize,
            betas=(args.adam_beta1, args.adam_beta2),
            eps=args.adam_epsilon,
            weight_decay=args.adam_weight_decay,
        )
    elif args.optimizer.lower() == "adam":
        optimizer_class = bnb.optim.Adam8bit if args.use_8bit_adam else torch.optim.Adam

        optimizer = optimizer_class(
            params_to_optimize,
            betas=(args.adam_beta1, args.adam_beta2),
            eps=args.adam_epsilon,
            weight_decay=args.adam_weight_decay,
        )
    elif args.optimizer.lower() == "prodigy":
        try:
            import prodigyopt
        except ImportError:
            raise ImportError("To use Prodigy, please install the prodigyopt library: `pip install prodigyopt`")

        optimizer_class = prodigyopt.Prodigy

        if args.learning_rate <= 0.1:
            logger.warning(
                "Learning rate is too low. When using prodigy, it's generally better to set learning rate around 1.0"
            )

        optimizer = optimizer_class(
            params_to_optimize,
            betas=(args.adam_beta1, args.adam_beta2),
            beta3=args.prodigy_beta3,
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
            decouple=args.prodigy_decouple,
            use_bias_correction=args.prodigy_use_bias_correction,
            safeguard_warmup=args.prodigy_safeguard_warmup,
        )

    return optimizer


'''
def main(args):
    if args.report_to == "wandb" and args.hub_token:
        raise ValueError("Cannot use both --report_to=wandb and --hub_token.")

    if torch.backends.mps.is_available() and args.mixed_precision == "bf16":
        raise ValueError("bf16 not supported on MPS; use fp16 or fp32.")

    logging_dir = Path(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    init_kwargs = InitProcessGroupKwargs(backend="nccl", timeout=timedelta(seconds=args.nccl_timeout))
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=[ddp_kwargs, init_kwargs],
    )

    if torch.backends.mps.is_available():
        accelerator.native_amp = False

    if args.report_to == "wandb" and not is_wandb_available():
        raise ImportError("Install wandb for logging: `pip install wandb`")

    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO)
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    if args.seed is not None:
        set_seed(args.seed)

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
        if args.push_to_hub:
            repo_id = create_repo(repo_id=args.hub_model_id or Path(args.output_dir).name, exist_ok=True).repo_id

    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer", revision=args.revision)
    text_encoder = T5EncoderModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision)
    load_dtype = torch.bfloat16 if "5b" in args.pretrained_model_name_or_path.lower() else torch.float16
    transformer = CogVideoXTransformer3DModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="transformer", torch_dtype=load_dtype, revision=args.revision, variant=args.variant)
    vae = AutoencoderKLCogVideoX.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision, variant=args.variant)
    scheduler = CogVideoXDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")

    if args.enable_slicing:
        vae.enable_slicing()
    if args.enable_tiling:
        vae.enable_tiling()

    text_encoder.requires_grad_(False)
    transformer.requires_grad_(False)
    vae.requires_grad_(False)

    weight_dtype = torch.float32
    if accelerator.state.deepspeed_plugin:
        if "fp16" in accelerator.state.deepspeed_plugin.deepspeed_config and accelerator.state.deepspeed_plugin.deepspeed_config["fp16"]["enabled"]:
            weight_dtype = torch.float16
        if "bf16" in accelerator.state.deepspeed_plugin.deepspeed_config and accelerator.state.deepspeed_plugin.deepspeed_config["bf16"]["enabled"]:
            weight_dtype = torch.bfloat16
    else:
        if accelerator.mixed_precision == "fp16":
            weight_dtype = torch.float16
        elif accelerator.mixed_precision == "bf16":
            weight_dtype = torch.bfloat16

    text_encoder.to(accelerator.device, dtype=weight_dtype)
    transformer.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)

    if args.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()

    transformer_lora_config = LoraConfig(r=args.rank, lora_alpha=args.lora_alpha, init_lora_weights=True, target_modules=["to_k", "to_q", "to_v", "to_out.0"])
    transformer.add_adapter(transformer_lora_config)

    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    def save_model_hook(models, weights, output_dir):
        if accelerator.is_main_process:
            transformer_lora_layers = None
            for model in models:
                if isinstance(model, type(unwrap_model(transformer))):
                    transformer_lora_layers = get_peft_model_state_dict(model)
                else:
                    raise ValueError(f"Unexpected save model: {model.__class__}")
                weights.pop()
            from diffusers import CogVideoXImageToVideoPipeline
            CogVideoXImageToVideoPipeline.save_lora_weights(output_dir, transformer_lora_layers=transformer_lora_layers)

    def load_model_hook(models, input_dir):
        transformer_ = None
        while len(models) > 0:
            model = models.pop()
            if isinstance(model, type(unwrap_model(transformer))):
                transformer_ = model
            else:
                raise ValueError(f"Unexpected save model: {model.__class__}")
        from diffusers import CogVideoXImageToVideoPipeline
        lora_state_dict = CogVideoXImageToVideoPipeline.lora_state_dict(input_dir)
        transformer_state_dict = {k.replace("transformer.", ""): v for k, v in lora_state_dict.items() if k.startswith("transformer.")}
        transformer_state_dict = convert_unet_state_dict_to_peft(transformer_state_dict)
        incompatible_keys = set_peft_model_state_dict(transformer_, transformer_state_dict, adapter_name="default")
        if incompatible_keys and getattr(incompatible_keys, "unexpected_keys", None):
            logger.warning(f"Unexpected keys in state_dict: {incompatible_keys.unexpected_keys}")

    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)

    if args.allow_tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes

    if args.mixed_precision == "fp16":
        cast_training_params([transformer], dtype=torch.float32)

    transformer_lora_parameters = list(filter(lambda p: p.requires_grad, transformer.parameters()))
    params_to_optimize = [{"params": transformer_lora_parameters, "lr": args.learning_rate}]

    use_deepspeed_optimizer = accelerator.state.deepspeed_plugin and "optimizer" in accelerator.state.deepspeed_plugin.deepspeed_config
    use_deepspeed_scheduler = accelerator.state.deepspeed_plugin and "scheduler" in accelerator.state.deepspeed_plugin.deepspeed_config
    optimizer = get_optimizer(args, params_to_optimize, use_deepspeed=use_deepspeed_optimizer)

    train_dataset = VideoDataset(
        instance_data_root=args.instance_data_root,
        dataset_name=args.dataset_name,
        dataset_config_name=args.dataset_config_name,
        caption_column=args.caption_column,
        input_video_column=args.input_video_column,
        target_video_column=args.target_video_column,
        height=args.height,
        width=args.width,
        fps=args.fps,
        max_num_frames=args.max_num_frames,
        skip_frames_start=args.skip_frames_start,
        skip_frames_end=args.skip_frames_end,
        cache_dir=args.cache_dir,
        id_token=args.id_token,
    )

    def encode_video_pair(input_video, target_video):
        input_video = input_video.to(accelerator.device, dtype=vae.dtype).unsqueeze(0).permute(0, 2, 1, 3, 4)
        target_video = target_video.to(accelerator.device, dtype=vae.dtype).unsqueeze(0).permute(0, 2, 1, 3, 4)
        
        input_noise_sigma = torch.normal(mean=-3.0, std=0.5, size=(1,), device=input_video.device)
        input_noise_sigma = torch.exp(input_noise_sigma).to(dtype=input_video.dtype)
        noisy_input_video = input_video + torch.randn_like(input_video) * input_noise_sigma[:, None, None, None, None]
        
        input_latent_dist = vae.encode(noisy_input_video).latent_dist
        target_latent_dist = vae.encode(target_video).latent_dist
        return input_latent_dist, target_latent_dist

    train_dataset.instance_prompts = [encode_prompt(tokenizer, text_encoder, prompt, 1, transformer.config.max_text_seq_length, accelerator.device, weight_dtype) for prompt in train_dataset.instance_prompts]
    train_dataset.instance_videos = [encode_video_pair(input_video, target_video) for input_video, target_video in zip(train_dataset.instance_input_videos, train_dataset.instance_target_videos)]

    def collate_fn(examples):
        input_videos, target_videos = [], []
        for example in examples:
            input_latent_dist, target_latent_dist = example["instance_video"]
            input_latents = input_latent_dist.sample() * vae.config.scaling_factor
            target_latents = target_latent_dist.sample() * vae.config.scaling_factor
            input_latents = input_latents.permute(0, 2, 1, 3, 4)
            target_latents = target_latents.permute(0, 2, 1, 3, 4)
            if random.random() < args.noised_video_dropout:
                input_latents = torch.zeros_like(input_latents)
            input_videos.append(input_latents)
            target_videos.append(target_latents)

        input_videos = torch.cat(input_videos)
        target_videos = torch.cat(target_videos)
        input_videos = input_videos.to(memory_format=torch.contiguous_format).float()
        target_videos = target_videos.to(memory_format=torch.contiguous_format).float()
        prompts = torch.cat([example["instance_prompt"] for example in examples])
        return {"videos": (input_videos, target_videos), "prompts": prompts}

    train_dataloader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True, collate_fn=collate_fn, num_workers=args.dataloader_num_workers)

    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    if use_deepspeed_scheduler:
        from accelerate.utils import DummyScheduler
        lr_scheduler = DummyScheduler(name=args.lr_scheduler, optimizer=optimizer, total_num_steps=args.max_train_steps * accelerator.num_processes, num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes)
    else:
        lr_scheduler = get_scheduler(args.lr_scheduler, optimizer=optimizer, num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes, num_training_steps=args.max_train_steps * accelerator.num_processes, num_cycles=args.lr_num_cycles, power=args.lr_power)

    transformer, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(transformer, optimizer, train_dataloader, lr_scheduler)

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    if accelerator.is_main_process:
        tracker_name = args.tracker_name or "cogvideox-v2v-lora"
        accelerator.init_trackers(tracker_name, config=vars(args))

    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    num_trainable_parameters = sum(param.numel() for model in params_to_optimize for param in model["params"])
    logger.info("***** Running training *****")
    logger.info(f"  Num trainable parameters = {num_trainable_parameters}")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size = {total_batch_size}")
    logger.info(f"  Gradient accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    if not args.resume_from_checkpoint:
        initial_global_step = 0
    else:
        path = os.path.basename(args.resume_from_checkpoint) if args.resume_from_checkpoint != "latest" else (sorted([d for d in os.listdir(args.output_dir) if d.startswith("checkpoint")], key=lambda x: int(x.split("-")[1]))[-1] if os.listdir(args.output_dir) else None)
        if path is None:
            accelerator.print(f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting new run.")
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])
            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch

    progress_bar = tqdm(range(0, args.max_train_steps), initial=initial_global_step, desc="Steps", disable=not accelerator.is_local_main_process)
    vae_scale_factor_spatial = 2 ** (len(vae.config.block_out_channels) - 1)

    model_config = transformer.module.config if hasattr(transformer, "module") else transformer.config

    for epoch in range(first_epoch, args.num_train_epochs):
        transformer.train()
        for step, batch in enumerate(train_dataloader):
            models_to_accumulate = [transformer]
            with accelerator.accumulate(models_to_accumulate):
                input_latents, target_latents = batch["videos"]
                prompt_embeds = batch["prompts"]

                input_latents = input_latents.to(dtype=weight_dtype)
                target_latents = target_latents.to(dtype=weight_dtype)

                batch_size, num_frames, num_channels, height, width = target_latents.shape
                timesteps = torch.randint(0, scheduler.config.num_train_timesteps, (batch_size,), device=target_latents.device).long()
                noise = torch.randn_like(target_latents)
                noisy_target_latents = scheduler.add_noise(target_latents, noise, timesteps)
                noisy_model_input = torch.cat([noisy_target_latents, input_latents], dim=2)

                image_rotary_emb = prepare_rotary_positional_embeddings(
                    height=args.height,
                    width=args.width,
                    num_frames=num_frames,
                    vae_scale_factor_spatial=vae_scale_factor_spatial,
                    patch_size=model_config.patch_size,
                    attention_head_dim=model_config.attention_head_dim,
                    device=accelerator.device,
                ) if model_config.use_rotary_positional_embeddings else None

                model_output = transformer(hidden_states=noisy_model_input, encoder_hidden_states=prompt_embeds, timestep=timesteps, image_rotary_emb=image_rotary_emb, return_dict=False)[0]
                model_pred = scheduler.get_velocity(model_output, noisy_target_latents, timesteps)

                alphas_cumprod = scheduler.alphas_cumprod[timesteps]
                weights = 1 / (1 - alphas_cumprod)
                while len(weights.shape) < len(model_pred.shape):
                    weights = weights.unsqueeze(-1)

                loss = torch.mean((weights * (model_pred - target_latents) ** 2).reshape(batch_size, -1), dim=1).mean()
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    params_to_clip = transformer.parameters()
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)

                if accelerator.state.deepspeed_plugin is None:
                    optimizer.step()
                    optimizer.zero_grad()

                lr_scheduler.step()

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                if accelerator.is_main_process or accelerator.distributed_type == DistributedType.DEEPSPEED:
                    if global_step % args.checkpointing_steps == 0:
                        if args.checkpoints_total_limit:
                            checkpoints = [d for d in os.listdir(args.output_dir) if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                for removing_checkpoint in checkpoints[:num_to_remove]:
                                    shutil.rmtree(os.path.join(args.output_dir, removing_checkpoint))
                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break

        if accelerator.is_main_process and args.validation_prompt and (epoch + 1) % args.validation_epochs == 0:
            from diffusers import CogVideoXImageToVideoPipeline
            pipe = CogVideoXImageToVideoPipeline.from_pretrained(args.pretrained_model_name_or_path, transformer=unwrap_model(transformer), scheduler=scheduler, revision=args.revision, variant=args.variant, torch_dtype=weight_dtype)
            validation_prompts = args.validation_prompt.split(args.validation_prompt_separator)
            validation_input_videos = args.validation_input_videos.split(args.validation_prompt_separator)

            for validation_input_video, validation_prompt in zip(validation_input_videos, validation_prompts):
                import decord
                vr = decord.VideoReader(validation_input_video, width=args.width, height=args.height)
                input_frames = vr.get_batch(list(range(min(len(vr), args.max_num_frames)))).float() / 255.0 * 2.0 - 1.0
                input_video = torch.stack([frame for frame in input_frames], dim=0).permute(0, 3, 1, 2).to(accelerator.device, dtype=weight_dtype).unsqueeze(0)
                pipeline_args = {"image": input_video[:, :, 0], "prompt": validation_prompt, "guidance_scale": args.guidance_scale, "use_dynamic_cfg": args.use_dynamic_cfg, "height": args.height, "width": args.width}
                log_validation(pipe=pipe, args=args, accelerator=accelerator, pipeline_args=pipeline_args, epoch=epoch)

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        transformer = unwrap_model(transformer)
        dtype = torch.float16 if args.mixed_precision == "fp16" else torch.bfloat16 if args.mixed_precision == "bf16" else torch.float32
        transformer = transformer.to(dtype)
        transformer_lora_layers = get_peft_model_state_dict(transformer)
        from diffusers import CogVideoXImageToVideoPipeline
        CogVideoXImageToVideoPipeline.save_lora_weights(save_directory=args.output_dir, transformer_lora_layers=transformer_lora_layers)
        del transformer
        free_memory()

        pipe = CogVideoXImageToVideoPipeline.from_pretrained(args.pretrained_model_name_or_path, revision=args.revision, variant=args.variant, torch_dtype=weight_dtype)
        pipe.scheduler = CogVideoXDPMScheduler.from_config(pipe.scheduler.config)
        if args.enable_slicing:
            pipe.vae.enable_slicing()
        if args.enable_tiling:
            pipe.vae.enable_tiling()
        lora_scaling = args.lora_alpha / args.rank
        pipe.load_lora_weights(args.output_dir, adapter_name="cogvideox-v2v-lora")
        pipe.set_adapters(["cogvideox-v2v-lora"], [lora_scaling])

        validation_outputs = []
        if args.validation_prompt and args.num_validation_videos > 0:
            validation_prompts = args.validation_prompt.split(args.validation_prompt_separator)
            validation_input_videos = args.validation_input_videos.split(args.validation_prompt_separator)
            for validation_input_video, validation_prompt in zip(validation_input_videos, validation_prompts):
                import decord
                vr = decord.VideoReader(validation_input_video, width=args.width, height=args.height)
                input_frames = vr.get_batch(list(range(min(len(vr), args.max_num_frames)))).float() / 255.0 * 2.0 - 1.0
                input_video = torch.stack([frame for frame in input_frames], dim=0).permute(0, 3, 1, 2).to(accelerator.device, dtype=weight_dtype).unsqueeze(0)
                pipeline_args = {"image": input_video[:, :, 0], "prompt": validation_prompt, "guidance_scale": args.guidance_scale, "use_dynamic_cfg": args.use_dynamic_cfg, "height": args.height, "width": args.width}
                video = log_validation(pipe=pipe, args=args, accelerator=accelerator, pipeline_args=pipeline_args, epoch=epoch, is_final_validation=True)
                validation_outputs.extend(video)

        if args.push_to_hub:
            validation_prompt = args.validation_prompt.split(args.validation_prompt_separator)[0] if args.validation_prompt else ""
            from diffusers.utils import save_model_card
            save_model_card(repo_id, videos=validation_outputs, base_model=args.pretrained_model_name_or_path, validation_prompt=validation_prompt, repo_folder=args.output_dir, fps=args.fps)
            upload_folder(repo_id=repo_id, folder_path=args.output_dir, commit_message="End of training", ignore_patterns=["step_*", "epoch_*"])

    accelerator.end_training()

if __name__ == "__main__":
    args = get_args()
    main(args)

'''


def main(args):
    # Enhanced error handling for incompatible options
    if args.report_to == "wandb" and args.hub_token:
        raise ValueError(
            "You cannot use both --report_to=wandb and --hub_token due to a security risk of exposing your Hugging Face token. "
            "Please authenticate with the Hub using `huggingface-cli login` instead."
        )

    if torch.backends.mps.is_available() and args.mixed_precision == "bf16":
        raise ValueError(
            "Mixed precision training with bfloat16 is not supported on MPS devices. "
            "Please use fp16 (recommended) or fp32 instead. Set --mixed_precision to 'fp16' or 'no'."
        )

    # Setup logging and accelerator
    logging_dir = Path(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    init_kwargs = InitProcessGroupKwargs(backend="nccl", timeout=timedelta(seconds=args.nccl_timeout))
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=[ddp_kwargs, init_kwargs],
    )

    if torch.backends.mps.is_available():
        accelerator.native_amp = False

    if args.report_to == "wandb" and not is_wandb_available():
        raise ImportError("WandB logging is enabled but not installed. Install it with `pip install wandb`.")

    # Configure verbose logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    if args.seed is not None:
        set_seed(args.seed)

    # Setup output directory and repository
    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name,
                exist_ok=True,
            ).repo_id

    # Load models
    logger.info("Loading tokenizer, text encoder, transformer, VAE, and scheduler...")
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer", revision=args.revision)
    text_encoder = T5EncoderModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision)
    load_dtype = torch.bfloat16 if "5b" in args.pretrained_model_name_or_path.lower() else torch.float16
    transformer = CogVideoXTransformer3DModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="transformer", torch_dtype=load_dtype, revision=args.revision, variant=args.variant)
    vae = AutoencoderKLCogVideoX.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision, variant=args.variant)
    scheduler = CogVideoXDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    logger.info("Models loaded successfully.")

    if args.enable_slicing:
        vae.enable_slicing()
    if args.enable_tiling:
        vae.enable_tiling()

    # Freeze non-trainable components
    text_encoder.requires_grad_(False)
    transformer.requires_grad_(False)
    vae.requires_grad_(False)

    # Determine weight dtype
    weight_dtype = torch.float32
    if accelerator.state.deepspeed_plugin:
        ds_config = accelerator.state.deepspeed_plugin.deepspeed_config
        if "fp16" in ds_config and ds_config["fp16"]["enabled"]:
            weight_dtype = torch.float16
        if "bf16" in ds_config and ds_config["bf16"]["enabled"]:
            weight_dtype = torch.bfloat16
    else:
        if accelerator.mixed_precision == "fp16":
            weight_dtype = torch.float16
        elif accelerator.mixed_precision == "bf16":
            weight_dtype = torch.bfloat16

    # Move models to device
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    transformer.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)

    if args.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()

    # Configure LoRA
    transformer_lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.lora_alpha,
        init_lora_weights=True,
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    )
    transformer.add_adapter(transformer_lora_config)

    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        return model._orig_mod if is_compiled_module(model) else model

    # Save and load hooks for LoRA weights
    def save_model_hook(models, weights, output_dir):
        if accelerator.is_main_process:
            transformer_lora_layers = None
            for model in models:
                if isinstance(model, type(unwrap_model(transformer))):
                    transformer_lora_layers = get_peft_model_state_dict(model)
                else:
                    raise ValueError(f"Unexpected model type in save hook: {model.__class__}")
                weights.pop()
            from diffusers import CogVideoXImageToVideoPipeline
            CogVideoXImageToVideoPipeline.save_lora_weights(output_dir, transformer_lora_layers=transformer_lora_layers)
            logger.info(f"Saved LoRA weights to {output_dir}")

    def load_model_hook(models, input_dir):
        transformer_ = None
        while models:
            model = models.pop()
            if isinstance(model, type(unwrap_model(transformer))):
                transformer_ = model
            else:
                raise ValueError(f"Unexpected model type in load hook: {model.__class__}")
        from diffusers import CogVideoXImageToVideoPipeline
        lora_state_dict = CogVideoXImageToVideoPipeline.lora_state_dict(input_dir)
        transformer_state_dict = {k.replace("transformer.", ""): v for k, v in lora_state_dict.items() if k.startswith("transformer.")}
        transformer_state_dict = convert_unet_state_dict_to_peft(transformer_state_dict)
        incompatible_keys = set_peft_model_state_dict(transformer_, transformer_state_dict, adapter_name="default")
        if incompatible_keys and getattr(incompatible_keys, "unexpected_keys", None):
            logger.warning(f"Unexpected keys in state_dict during load: {incompatible_keys.unexpected_keys}")

    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)

    if args.allow_tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes

    if args.mixed_precision == "fp16":
        cast_training_params([transformer], dtype=torch.float32)

    # Setup optimizer
    transformer_lora_parameters = list(filter(lambda p: p.requires_grad, transformer.parameters()))
    params_to_optimize = [{"params": transformer_lora_parameters, "lr": args.learning_rate}]
    optimizer = get_optimizer(args, params_to_optimize, use_deepspeed=use_deepspeed_optimizer)

    # Load and validate V2V dataset
    logger.info("Loading V2V dataset...")
    train_dataset = VideoDataset(
        instance_data_root=args.instance_data_root,
        dataset_name=args.dataset_name,
        dataset_config_name=args.dataset_config_name,
        caption_column=args.caption_column,
        input_video_column=args.input_video_column,
        target_video_column=args.target_video_column,
        height=args.height,
        width=args.width,
        fps=args.fps,
        max_num_frames=args.max_num_frames,
        skip_frames_start=args.skip_frames_start,
        skip_frames_end=args.skip_frames_end,
        cache_dir=args.cache_dir,
        id_token=args.id_token,
    )
    if len(train_dataset.instance_input_videos) != len(train_dataset.instance_target_videos):
        raise ValueError(f"Mismatch in dataset: {len(train_dataset.instance_input_videos)} input videos vs {len(train_dataset.instance_target_videos)} target videos.")
    logger.info(f"Loaded {len(train_dataset)} video pairs for training.")

    # V2V-specific encoding
    def encode_video_pair(input_video, target_video):
        logger.debug("Encoding video pair...")
        if input_video.shape != target_video.shape:
            raise ValueError("Input and target videos must have the same dimensions.")
        input_video = input_video.to(accelerator.device, dtype=vae.dtype).unsqueeze(0).permute(0, 2, 1, 3, 4)
        target_video = target_video.to(accelerator.device, dtype=vae.dtype).unsqueeze(0).permute(0, 2, 1, 3, 4)
        input_noise_sigma = torch.exp(torch.normal(mean=-3.0, std=0.5, size=(1,), device=input_video.device)).to(dtype=input_video.dtype)
        noisy_input_video = input_video + torch.randn_like(input_video) * input_noise_sigma[:, None, None, None, None]
        input_latent_dist = vae.encode(noisy_input_video).latent_dist
        target_latent_dist = vae.encode(target_video).latent_dist
        logger.debug("Encoding complete.")
        return input_latent_dist, target_latent_dist

    train_dataset.instance_prompts = [encode_prompt(tokenizer, text_encoder, prompt, 1, transformer.config.max_text_seq_length, accelerator.device, weight_dtype) for prompt in train_dataset.instance_prompts]
    train_dataset.instance_videos = [encode_video_pair(input_video, target_video) for input_video, target_video in zip(train_dataset.instance_input_videos, train_dataset.instance_target_videos)]

    # Collate function with shape validation
    def collate_fn(examples):
        input_videos, target_videos = [], []
        for example in examples:
            input_latent_dist, target_latent_dist = example["instance_video"]
            input_latents = input_latent_dist.sample() * vae.config.scaling_factor
            target_latents = target_latent_dist.sample() * vae.config.scaling_factor
            input_latents = input_latents.permute(0, 2, 1, 3, 4)
            target_latents = target_latents.permute(0, 2, 1, 3, 4)
            assert input_latents.shape == target_latents.shape, "Input and target latents must have identical shapes."
            if random.random() < args.noised_video_dropout:
                input_latents = torch.zeros_like(input_latents)
            input_videos.append(input_latents)
            target_videos.append(target_latents)
        input_videos = torch.cat(input_videos).to(memory_format=torch.contiguous_format).float()
        target_videos = torch.cat(target_videos).to(memory_format=torch.contiguous_format).float()
        prompts = torch.cat([example["instance_prompt"] for example in examples])
        return {"videos": (input_videos, target_videos), "prompts": prompts}

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=args.dataloader_num_workers,
    )

    # Setup scheduler
    use_deepspeed_optimizer = accelerator.state.deepspeed_plugin and "optimizer" in accelerator.state.deepspeed_plugin.deepspeed_config
    use_deepspeed_scheduler = accelerator.state.deepspeed_plugin and "scheduler" in accelerator.state.deepspeed_plugin.deepspeed_config
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    if use_deepspeed_scheduler:
        from accelerate.utils import DummyScheduler
        lr_scheduler = DummyScheduler(
            name=args.lr_scheduler,
            optimizer=optimizer,
            total_num_steps=args.max_train_steps * accelerator.num_processes,
            num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        )
    else:
        lr_scheduler = get_scheduler(
            args.lr_scheduler,
            optimizer=optimizer,
            num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
            num_training_steps=args.max_train_steps * accelerator.num_processes,
            num_cycles=args.lr_num_cycles,
            power=args.lr_power,
        )

    transformer, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(transformer, optimizer, train_dataloader, lr_scheduler)

    # Recalculate training steps
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # Initialize trackers
    if accelerator.is_main_process:
        tracker_name = args.tracker_name or "cogvideox-v2v-lora"
        accelerator.init_trackers(tracker_name, config=vars(args))

    # Log training configuration
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    num_trainable_parameters = sum(param.numel() for model in params_to_optimize for param in model["params"])
    logger.info("***** Running V2V Training *****")
    logger.info(f"  Num trainable parameters = {num_trainable_parameters}")
    logger.info(f"  Num video pairs = {len(train_dataset)}")
    logger.info(f"  Num batches per epoch = {len(train_dataloader)}")
    logger.info(f"  Num epochs = {args.num_train_epochs}")
    logger.info(f"  Batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size = {total_batch_size}")
    logger.info(f"  Gradient accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")

    # Training loop
    global_step = 0
    first_epoch = 0
    if args.resume_from_checkpoint:
        path = os.path.basename(args.resume_from_checkpoint) if args.resume_from_checkpoint != "latest" else (sorted([d for d in os.listdir(args.output_dir) if d.startswith("checkpoint")], key=lambda x: int(x.split("-")[1]))[-1] if os.listdir(args.output_dir) else None)
        if path is None:
            logger.info(f"Checkpoint '{args.resume_from_checkpoint}' not found. Starting fresh.")
            initial_global_step = 0
        else:
            logger.info(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])
            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch
    else:
        initial_global_step = 0

    progress_bar = tqdm(range(0, args.max_train_steps), initial=initial_global_step, desc="Steps", disable=not accelerator.is_local_main_process)
    vae_scale_factor_spatial = 2 ** (len(vae.config.block_out_channels) - 1)
    model_config = transformer.module.config if hasattr(transformer, "module") else transformer.config

    for epoch in range(first_epoch, args.num_train_epochs):
        logger.info(f"Starting epoch {epoch + 1}/{args.num_train_epochs}")
        transformer.train()
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate([transformer]):
                input_latents, target_latents = batch["videos"]
                prompt_embeds = batch["prompts"]
                input_latents = input_latents.to(dtype=weight_dtype)
                target_latents = target_latents.to(dtype=weight_dtype)

                batch_size, num_frames, num_channels, height, width = target_latents.shape
                timesteps = torch.randint(0, scheduler.config.num_train_timesteps, (batch_size,), device=target_latents.device).long()
                noise = torch.randn_like(target_latents)
                noisy_target_latents = scheduler.add_noise(target_latents, noise, timesteps)
                # V2V-specific: Concatenate input and noisy target latents
                noisy_model_input = torch.cat([noisy_target_latents, input_latents], dim=2)

                image_rotary_emb = prepare_rotary_positional_embeddings(
                    height=args.height,
                    width=args.width,
                    num_frames=num_frames,
                    vae_scale_factor_spatial=vae_scale_factor_spatial,
                    patch_size=model_config.patch_size,
                    attention_head_dim=model_config.attention_head_dim,
                    device=accelerator.device,
                ) if model_config.use_rotary_positional_embeddings else None

                model_output = transformer(hidden_states=noisy_model_input, encoder_hidden_states=prompt_embeds, timestep=timesteps, image_rotary_emb=image_rotary_emb, return_dict=False)[0]
                model_pred = scheduler.get_velocity(model_output, noisy_target_latents, timesteps)

                alphas_cumprod = scheduler.alphas_cumprod[timesteps]
                weights = 1 / (1 - alphas_cumprod)
                while len(weights.shape) < len(model_pred.shape):
                    weights = weights.unsqueeze(-1)

                loss = torch.mean((weights * (model_pred - target_latents) ** 2).reshape(batch_size, -1), dim=1).mean()
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(transformer.parameters(), args.max_grad_norm)
                    optimizer.step()
                    optimizer.zero_grad()
                    lr_scheduler.step()

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                if (accelerator.is_main_process or accelerator.distributed_type == DistributedType.DEEPSPEED) and global_step % args.checkpointing_steps == 0:
                    if args.checkpoints_total_limit:
                        checkpoints = sorted([d for d in os.listdir(args.output_dir) if d.startswith("checkpoint")], key=lambda x: int(x.split("-")[1]))
                        if len(checkpoints) >= args.checkpoints_total_limit:
                            for checkpoint in checkpoints[:len(checkpoints) - args.checkpoints_total_limit + 1]:
                                shutil.rmtree(os.path.join(args.output_dir, checkpoint))
                    save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    accelerator.save_state(save_path)
                    logger.info(f"Saved checkpoint at step {global_step} to {save_path}")

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
            if step % 100 == 0:
                logger.info(f"Epoch {epoch + 1}, Step {step}, Loss: {loss.item():.4f}")

            if global_step >= args.max_train_steps:
                break

        logger.info(f"Completed epoch {epoch + 1}")

        # Validation with V2V-specific workaround
        if accelerator.is_main_process and args.validation_prompt and (epoch + 1) % args.validation_epochs == 0:
            logger.warning(
                "Using first frame of input video as conditioning for validation due to lack of dedicated V2V pipeline."
            )
            from diffusers import CogVideoXImageToVideoPipeline
            pipe = CogVideoXImageToVideoPipeline.from_pretrained(args.pretrained_model_name_or_path, transformer=unwrap_model(transformer), scheduler=scheduler, revision=args.revision, variant=args.variant, torch_dtype=weight_dtype)
            validation_prompts = args.validation_prompt.split(args.validation_prompt_separator)
            validation_input_videos = args.validation_input_videos.split(args.validation_prompt_separator)
            for vid_path, prompt in zip(validation_input_videos, validation_prompts):
                if not os.path.exists(vid_path):
                    logger.error(f"Validation video {vid_path} not found.")
                    continue
                import decord
                vr = decord.VideoReader(vid_path, width=args.width, height=args.height)
                input_frames = vr.get_batch(list(range(min(len(vr), args.max_num_frames)))).float() / 255.0 * 2.0 - 1.0
                input_video = torch.stack([frame for frame in input_frames], dim=0).permute(0, 3, 1, 2).to(accelerator.device, dtype=weight_dtype).unsqueeze(0)
                pipeline_args = {"image": input_video[:, :, 0], "prompt": prompt, "guidance_scale": args.guidance_scale, "use_dynamic_cfg": args.use_dynamic_cfg, "height": args.height, "width": args.width}
                log_validation(pipe=pipe, args=args, accelerator=accelerator, pipeline_args=pipeline_args, epoch=epoch)

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        logger.info("Saving final LoRA weights...")
        transformer = unwrap_model(transformer).to(torch.float16 if args.mixed_precision == "fp16" else torch.bfloat16 if args.mixed_precision == "bf16" else torch.float32)
        transformer_lora_layers = get_peft_model_state_dict(transformer)
        from diffusers import CogVideoXImageToVideoPipeline
        CogVideoXImageToVideoPipeline.save_lora_weights(save_directory=args.output_dir, transformer_lora_layers=transformer_lora_layers)
        logger.info("LoRA weights saved.")
        del transformer
        free_memory()

        logger.info("Performing final validation...")
        pipe = CogVideoXImageToVideoPipeline.from_pretrained(args.pretrained_model_name_or_path, revision=args.revision, variant=args.variant, torch_dtype=weight_dtype)
        pipe.scheduler = CogVideoXDPMScheduler.from_config(pipe.scheduler.config)
        if args.enable_slicing:
            pipe.vae.enable_slicing()
        if args.enable_tiling:
            pipe.vae.enable_tiling()
        lora_scaling = args.lora_alpha / args.rank
        pipe.load_lora_weights(args.output_dir, adapter_name="cogvideox-v2v-lora")
        pipe.set_adapters(["cogvideox-v2v-lora"], [lora_scaling])

        validation_outputs = []
        if args.validation_prompt and args.num_validation_videos > 0:
            validation_prompts = args.validation_prompt.split(args.validation_prompt_separator)
            validation_input_videos = args.validation_input_videos.split(args.validation_prompt_separator)
            for vid_path, prompt in zip(validation_input_videos, validation_prompts):
                if not os.path.exists(vid_path):
                    logger.error(f"Final validation video {vid_path} not found.")
                    continue
                import decord
                vr = decord.VideoReader(vid_path, width=args.width, height=args.height)
                input_frames = vr.get_batch(list(range(min(len(vr), args.max_num_frames)))).float() / 255.0 * 2.0 - 1.0
                input_video = torch.stack([frame for frame in input_frames], dim=0).permute(0, 3, 1, 2).to(accelerator.device, dtype=weight_dtype).unsqueeze(0)
                pipeline_args = {"image": input_video[:, :, 0], "prompt": prompt, "guidance_scale": args.guidance_scale, "use_dynamic_cfg": args.use_dynamic_cfg, "height": args.height, "width": args.width}
                video = log_validation(pipe=pipe, args=args, accelerator=accelerator, pipeline_args=pipeline_args, epoch=epoch, is_final_validation=True)
                validation_outputs.extend(video)
        logger.info("Final validation complete.")

        if args.push_to_hub:
            validation_prompt = args.validation_prompt.split(args.validation_prompt_separator)[0] if args.validation_prompt else ""
            from diffusers.utils import save_model_card
            save_model_card(repo_id, videos=validation_outputs, base_model=args.pretrained_model_name_or_path, validation_prompt=validation_prompt, repo_folder=args.output_dir, fps=args.fps)
            upload_folder(repo_id=repo_id, folder_path=args.output_dir, commit_message="End of training", ignore_patterns=["step_*", "epoch_*"])

    accelerator.end_training()

if __name__ == "__main__":
    args = get_args()
    main(args)
