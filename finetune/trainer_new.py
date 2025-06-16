import hashlib
import json
import logging
import math
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, List, Tuple

import diffusers
import torch
import transformers
import wandb
from accelerate.accelerator import Accelerator, DistributedType
from accelerate.logging import get_logger
from accelerate.utils import (
    DistributedDataParallelKwargs,
    InitProcessGroupKwargs,
    ProjectConfiguration,
    gather_object,
    set_seed,
)
from diffusers import AutoencoderKLTemporalDecoder, UNetSpatioTemporalConditionModel
from diffusers.optimization import get_scheduler
from diffusers.pipelines import StableVideoDiffusionPipeline
from diffusers.utils.export_utils import export_to_video
from peft import LoraConfig, get_peft_model_state_dict, set_peft_model_state_dict
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

from finetune.constants import LOG_LEVEL, LOG_NAME
from finetune.datasets import I2VDatasetWithResize
from finetune.datasets.utils import (
    load_prompts,
    load_videos,
    preprocess_video_with_resize,
)
from finetune.schemas import Args, Components, State
from finetune.utils import (
    cast_training_params,
    free_memory,
    get_intermediate_ckpt_path,
    get_latest_ckpt_path_to_resume_from,
    get_memory_statistics,
    get_optimizer,
    string_to_filename,
    unload_model,
    unwrap_model,
)

logger = get_logger(LOG_NAME, LOG_LEVEL)

_DTYPE_MAP = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}

class Trainer:
    UNLOAD_LIST: List[str] = None

    def __init__(self, args: Args) -> None:
        self.args = args
        self.state = State(
            weight_dtype=self.__get_training_dtype(),
            train_frames=self.args.train_resolution[0],
            train_height=self.args.train_resolution[1],
            train_width=self.args.train_resolution[2],
        )
        self.components: Components = self.load_components()
        self.accelerator: Accelerator = None
        self.dataset: Dataset = None
        self.data_loader: DataLoader = None
        self.optimizer = None
        self.lr_scheduler = None
        self._init_distributed()
        self._init_logging()
        self._init_directories()
        self.state.using_deepspeed = self.accelerator.state.deepspeed_plugin is not None

    def _init_distributed(self):
        logging_dir = Path(self.args.output_dir, "logs")
        project_config = ProjectConfiguration(project_dir=self.args.output_dir, logging_dir=logging_dir)
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        init_process_group_kwargs = InitProcessGroupKwargs(backend="nccl", timeout=timedelta(seconds=self.args.nccl_timeout))
        mixed_precision = "no" if torch.backends.mps.is_available() else self.args.mixed_precision
        report_to = None if self.args.report_to.lower() == "none" else self.args.report_to
        accelerator = Accelerator(
            project_config=project_config,
            gradient_accumulation_steps=self.args.gradient_accumulation_steps,
            mixed_precision=mixed_precision,
            log_with=report_to,
            kwargs_handlers=[ddp_kwargs, init_process_group_kwargs],
        )
        if torch.backends.mps.is_available():
            accelerator.native_amp = False
        self.accelerator = accelerator
        if self.args.seed is not None:
            set_seed(self.args.seed)

    def _init_logging(self) -> None:
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=LOG_LEVEL,
        )
        if self.accelerator.is_local_main_process:
            transformers.utils.logging.set_verbosity_warning()
            diffusers.utils.logging.set_verbosity_info()
        else:
            transformers.utils.logging.set_verbosity_error()
            diffusers.utils.logging.set_verbosity_error()
        logger.info("Initialized Trainer")
        logger.info(f"Accelerator state: \n{self.accelerator.state}", main_process_only=False)

    def _init_directories(self) -> None:
        if self.accelerator.is_main_process:
            self.args.output_dir = Path(self.args.output_dir)
            self.args.output_dir.mkdir(parents=True, exist_ok=True)

    def check_setting(self) -> None:
        if self.UNLOAD_LIST is None:
            logger.warning("\033[91mNo unload_list specified. All components will be loaded to GPU.\033[0m")
        else:
            for name in self.UNLOAD_LIST:
                if name not in self.components.model_fields:
                    raise ValueError(f"Invalid component name in unload_list: {name}")

    def prepare_models(self) -> None:
        logger.info("Initializing models")
        if self.components.vae is not None:
            if self.args.enable_slicing:
                self.components.vae.enable_slicing()
            if self.args.enable_tiling:
                self.components.vae.enable_tiling()
        self.state.transformer_config = self.components.transformer.config

    def prepare_dataset(self) -> None:
        logger.info("Initializing dataset and dataloader")
        dataset_args = self.args.model_dump()
        dataset_args["gt_video_column"] = self.args.video_column
        self.dataset = I2VDatasetWithResize(
            **dataset_args,
            device=self.accelerator.device,
            max_num_frames=self.state.train_frames,
            height=self.state.train_height,
            width=self.state.train_width,
            trainer=self,
        )
        self.components.vae.requires_grad_(False)
        self.components.text_encoder.requires_grad_(False)
        self.components.vae = self.components.vae.to(self.accelerator.device, dtype=self.state.weight_dtype)
        self.components.text_encoder = self.components.text_encoder.to(self.accelerator.device, dtype=self.state.weight_dtype)
        logger.info("Precomputing latent for video and prompt embedding ...")
        tmp_data_loader = DataLoader(self.dataset, collate_fn=self.collate_fn, batch_size=1, num_workers=0, pin_memory=False)
        tmp_data_loader = self.accelerator.prepare_data_loader(tmp_data_loader)
        for _ in tmp_data_loader:
            pass
        self.accelerator.wait_for_everyone()
        logger.info("Precomputing latent for video and prompt embedding ... Done")
        unload_model(self.components.vae)
        unload_model(self.components.text_encoder)
        free_memory()
        self.data_loader = DataLoader(
            self.dataset,
            collate_fn=self.collate_fn,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            pin_memory=self.args.pin_memory,
            shuffle=True,
        )

    def prepare_trainable_parameters(self):
        logger.info("Initializing trainable parameters")
        weight_dtype = self.state.weight_dtype
        if torch.backends.mps.is_available() and weight_dtype == torch.bfloat16:
            raise ValueError("bf16 not supported on MPS. Use fp16 or fp32.")
        for attr_name, component in vars(self.components).items():
            if hasattr(component, "requires_grad_"):
                if self.args.training_type == "sft" and attr_name == "transformer":
                    component.requires_grad_(True)
                else:
                    component.requires_grad_(False)
        if self.args.training_type == "lora":
            transformer_lora_config = LoraConfig(
                r=self.args.rank,
                lora_alpha=self.args.lora_alpha,
                init_lora_weights=True,
                target_modules=self.args.target_modules,
            )
            self.components.transformer.add_adapter(transformer_lora_config)
            self.__prepare_saving_loading_hooks(transformer_lora_config)
        ignore_list = ["transformer"] + self.UNLOAD_LIST
        self.__move_components_to_device(dtype=weight_dtype, ignore_list=ignore_list)
        if self.args.gradient_checkpointing:
            self.components.transformer.enable_gradient_checkpointing()

    def prepare_optimizer(self) -> None:
        logger.info("Initializing optimizer and lr scheduler")
        cast_training_params([self.components.transformer], dtype=torch.float32)
        trainable_parameters = list(filter(lambda p: p.requires_grad, self.components.transformer.parameters()))
        transformer_parameters_with_lr = {"params": trainable_parameters, "lr": self.args.learning_rate}
        params_to_optimize = [transformer_parameters_with_lr]
        self.state.num_trainable_parameters = sum(p.numel() for p in trainable_parameters)
        use_deepspeed_opt = self.accelerator.state.deepspeed_plugin is not None and "optimizer" in self.accelerator.state.deepspeed_plugin.deepspeed_config
        optimizer = get_optimizer(
            params_to_optimize=params_to_optimize,
            optimizer_name=self.args.optimizer,
            learning_rate=self.args.learning_rate,
            beta1=self.args.beta1,
            beta2=self.args.beta2,
            beta3=self.args.beta3,
            epsilon=self.args.epsilon,
            weight_decay=self.args.weight_decay,
            use_deepspeed=use_deepspeed_opt,
        )
        num_update_steps_per_epoch = math.ceil(len(self.data_loader) / self.args.gradient_accumulation_steps)
        if self.args.train_steps is None:
            self.args.train_steps = self.args.train_epochs * num_update_steps_per_epoch
            self.state.overwrote_max_train_steps = True
        use_deepspeed_lr_scheduler = self.accelerator.state.deepspeed_plugin is not None and "scheduler" in self.accelerator.state.deepspeed_plugin.deepspeed_config
        total_training_steps = self.args.train_steps * self.accelerator.num_processes
        num_warmup_steps = self.args.lr_warmup_steps * self.accelerator.num_processes
        if use_deepspeed_lr_scheduler:
            from accelerate.utils import DummyScheduler
            lr_scheduler = DummyScheduler(name=self.args.lr_scheduler, optimizer=optimizer, total_num_steps=total_training_steps, num_warmup_steps=num_warmup_steps)
        else:
            lr_scheduler = get_scheduler(
                name=self.args.lr_scheduler,
                optimizer=optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=total_training_steps,
                num_cycles=self.args.lr_num_cycles,
                power=self.args.lr_power,
            )
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

    def prepare_for_training(self) -> None:
        self.components.transformer, self.optimizer, self.data_loader, self.lr_scheduler = self.accelerator.prepare(self.components.transformer, self.optimizer, self.data_loader, self.lr_scheduler)
        num_update_steps_per_epoch = math.ceil(len(self.data_loader) / self.args.gradient_accumulation_steps)
        if self.state.overwrote_max_train_steps:
            self.args.train_steps = self.args.train_epochs * num_update_steps_per_epoch
        self.args.train_epochs = math.ceil(self.args.train_steps / num_update_steps_per_epoch)
        self.state.num_update_steps_per_epoch = num_update_steps_per_epoch

    def prepare_for_validation(self):
        validation_prompts = load_prompts(self.args.validation_dir / self.args.validation_prompts)
        if self.args.validation_videos is not None:
            validation_input_videos = load_videos(self.args.validation_dir / self.args.validation_videos)
        else:
            validation_input_videos = [None] * len(validation_prompts)
        self.state.validation_prompts = validation_prompts
        self.state.validation_input_videos = validation_input_videos

    def prepare_trackers(self) -> None:
        logger.info("Initializing trackers")
        tracker_name = self.args.tracker_name or "finetrainers-experiment"
        self.accelerator.init_trackers(tracker_name, config=self.args.model_dump())

    def train(self) -> None:
        logger.info("Starting training")
        memory_statistics = get_memory_statistics()
        logger.info(f"Memory before training start: {json.dumps(memory_statistics, indent=4)}")
        self.state.total_batch_size_count = self.args.batch_size * self.accelerator.num_processes * self.args.gradient_accumulation_steps
        info = {
            "trainable parameters": self.state.num_trainable_parameters,
            "total samples": len(self.dataset),
            "train epochs": self.args.train_epochs,
            "train steps": self.args.train_steps,
            "batches per device": self.args.batch_size,
            "total batches observed per epoch": len(self.data_loader),
            "train batch size total count": self.state.total_batch_size_count,
            "gradient accumulation steps": self.args.gradient_accumulation_steps,
        }
        logger.info(f"Training configuration: {json.dumps(info, indent=4)}")
        global_step = 0
        first_epoch = 0
        initial_global_step = 0
        resume_from_checkpoint_path, initial_global_step, global_step, first_epoch = get_latest_ckpt_path_to_resume_from(
            resume_from_checkpoint=self.args.resume_from_checkpoint,
            num_update_steps_per_epoch=self.state.num_update_steps_per_epoch,
        )
        if resume_from_checkpoint_path is not None:
            self.accelerator.load_state(resume_from_checkpoint_path)
        progress_bar = tqdm(range(0, self.args.train_steps), initial=initial_global_step, desc="Training steps", disable=not self.accelerator.is_local_main_process)
        accelerator = self.accelerator
        generator = torch.Generator(device=accelerator.device)
        if self.args.seed is not None:
            generator = generator.manual_seed(self.args.seed)
        self.state.generator = generator
        free_memory()
        for epoch in range(first_epoch, self.args.train_epochs):
            logger.debug(f"Starting epoch ({epoch + 1}/{self.args.train_epochs})")
            self.components.transformer.train()
            models_to_accumulate = [self.components.transformer]
            for step, batch in enumerate(self.data_loader):
                logger.debug(f"Starting step {step + 1}")
                logs = {}
                with accelerator.accumulate(models_to_accumulate):
                    loss = self.compute_loss(batch)
                    accelerator.backward(loss)
                    if accelerator.sync_gradients:
                        if accelerator.distributed_type == DistributedType.DEEPSPEED:
                            grad_norm = self.components.transformer.get_global_grad_norm()
                            if torch.is_tensor(grad_norm):
                                grad_norm = grad_norm.item()
                        else:
                            grad_norm = accelerator.clip_grad_norm_(self.components.transformer.parameters(), self.args.max_grad_norm)
                            if torch.is_tensor(grad_norm):
                                grad_norm = grad_norm.item()
                        logs["grad_norm"] = grad_norm
                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad()
                if accelerator.sync_gradients:
                    progress_bar.update(1)
                    global_step += 1
                    self.__maybe_save_checkpoint(global_step)
                logs["loss"] = loss.detach().item()
                logs["lr"] = self.lr_scheduler.get_last_lr()[0]
                progress_bar.set_postfix(logs)
                should_run_validation = self.args.do_validation and global_step % self.args.validation_steps == 0
                if should_run_validation:
                    del loss
                    free_memory()
                    self.validate(global_step)
                accelerator.log(logs, step=global_step)
                if global_step >= self.args.train_steps:
                    break
            memory_statistics = get_memory_statistics()
            logger.info(f"Memory after epoch {epoch + 1}: {json.dumps(memory_statistics, indent=4)}")
        accelerator.wait_for_everyone()
        self.__maybe_save_checkpoint(global_step, must_save=True)
        if self.args.do_validation:
            free_memory()
            self.validate(global_step)
        del self.components
        free_memory()
        memory_statistics = get_memory_statistics()
        logger.info(f"Memory after training end: {json.dumps(memory_statistics, indent=4)}")
        accelerator.end_training()

    def validate(self, step: int) -> None:
        logger.info("Starting validation")
        accelerator = self.accelerator
        num_validation_samples = len(self.state.validation_prompts)
        if num_validation_samples == 0:
            logger.warning("No validation samples found. Skipping validation.")
            return
        self.components.transformer.eval()
        torch.set_grad_enabled(False)
        memory_statistics = get_memory_statistics()
        logger.info(f"Memory before validation start: {json.dumps(memory_statistics, indent=4)}")
        pipe = self.initialize_pipeline()
        if self.state.using_deepspeed:
            self.__move_components_to_device(dtype=self.state.weight_dtype, ignore_list=["transformer"])
        else:
            pipe.enable_model_cpu_offload(device=self.accelerator.device)
            pipe = pipe.to(dtype=self.state.weight_dtype)
        all_processes_artifacts = []
        for i in range(num_validation_samples):
            if self.state.using_deepspeed and self.accelerator.deepspeed_plugin.zero_stage != 3:
                if i % accelerator.num_processes != accelerator.process_index:
                    continue
            prompt = self.state.validation_prompts[i]
            input_video_path = self.state.validation_input_videos[i]
            if input_video_path is not None:
                input_video = preprocess_video_with_resize(input_video_path, self.state.train_frames, self.state.train_height, self.state.train_width)
                input_video = input_video.to(self.accelerator.device, dtype=self.state.weight_dtype).unsqueeze(0)
                # Use first frame as conditioning image for compatibility with SVD pipeline
                input_first_frame = input_video[:, :, 0, :, :]
            else:
                input_first_frame = None
            logger.debug(f"Validating sample {i + 1}/{num_validation_samples} on process {accelerator.process_index}. Prompt: {prompt}", main_process_only=False)
            validation_artifacts = self.validation_step({"prompt": prompt, "image": input_first_frame}, pipe)
            if self.state.using_deepspeed and self.accelerator.deepspeed_plugin.zero_stage == 3 and not accelerator.is_main_process:
                continue
            prompt_filename = string_to_filename(prompt)[:25]
            reversed_prompt = prompt[::-1]
            hash_suffix = hashlib.md5(reversed_prompt.encode()).hexdigest()[:5]
            artifacts = {"video": {"type": "video", "value": validation_artifacts}}
            for key, value in artifacts.items():
                artifact_type = value["type"]
                artifact_value = value["value"]
                if artifact_type != "video" or artifact_value is None:
                    continue
                filename = f"validation-{step}-{accelerator.process_index}-{prompt_filename}-{hash_suffix}.mp4"
                validation_path = self.args.output_dir / "validation_res"
                validation_path.mkdir(parents=True, exist_ok=True)
                filename = str(validation_path / filename)
                logger.debug(f"Saving video to {filename}")
                export_to_video(artifact_value, filename, fps=self.args.gen_fps)
                artifact_value = wandb.Video(filename, caption=prompt)
                all_processes_artifacts.append(artifact_value)
        all_artifacts = gather_object(all_processes_artifacts)
        if accelerator.is_main_process:
            tracker_key = "validation"
            for tracker in accelerator.trackers:
                if tracker.name == "wandb":
                    video_artifacts = [artifact for artifact in all_artifacts if isinstance(artifact, wandb.Video)]
                    tracker.log({tracker_key: {"videos": video_artifacts}}, step=step)
        if self.state.using_deepspeed:
            del pipe
            self.__move_components_to_cpu(unload_list=self.UNLOAD_LIST)
        else:
            pipe.remove_all_hooks()
            del pipe
            self.__move_components_to_device(dtype=self.state.weight_dtype, ignore_list=self.UNLOAD_LIST)
            self.components.transformer.to(self.accelerator.device, dtype=self.state.weight_dtype)
            cast_training_params([self.components.transformer], dtype=torch.float32)
        free_memory()
        accelerator.wait_for_everyone()
        memory_statistics = get_memory_statistics()
        logger.info(f"Memory after validation end: {json.dumps(memory_statistics, indent=4)}")
        torch.cuda.reset_peak_memory_stats(accelerator.device)
        torch.set_grad_enabled(True)
        self.components.transformer.train()

    def fit(self):
        self.check_setting()
        self.prepare_models()
        self.prepare_dataset()
        self.prepare_trainable_parameters()
        self.prepare_optimizer()
        self.prepare_for_training()
        if self.args.do_validation:
            self.prepare_for_validation()
        self.prepare_trackers()
        self.train()

    def collate_fn(self, examples: List[Dict[str, Any]]):
        input_videos = []
        target_videos = []
        prompt_embeddings = []
        for example in examples:
            input_video = example["input_video"].to(self.accelerator.device, dtype=self.state.weight_dtype)
            target_video = example["encoded_video"].to(self.accelerator.device, dtype=self.state.weight_dtype)
            prompt_embedding = example["prompt_embedding"].to(self.accelerator.device, dtype=self.state.weight_dtype)
            input_videos.append(input_video)
            target_videos.append(target_video)
            prompt_embeddings.append(prompt_embedding)
        input_videos = torch.stack(input_videos)
        target_videos = torch.stack(target_videos)
        prompt_embeddings = torch.stack(prompt_embeddings)
        return {"input_videos": input_videos, "target_videos": target_videos, "prompt_embeddings": prompt_embeddings}

    def load_components(self) -> Components:
        logger.info("Loading model components for V2V")
        vae = AutoencoderKLTemporalDecoder.from_pretrained(self.args.model_path, subfolder="vae")
        text_encoder = CLIPTextModel.from_pretrained(self.args.model_path, subfolder="text_encoder")
        tokenizer = CLIPTokenizer.from_pretrained(self.args.model_path, subfolder="tokenizer")
        transformer = UNetSpatioTemporalConditionModel.from_pretrained(self.args.model_path, subfolder="transformer")
        return Components(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            transformer=transformer,
            pipeline_cls=StableVideoDiffusionPipeline,
        )

    def initialize_pipeline(self) -> StableVideoDiffusionPipeline:
        logger.info("Initializing pipeline for V2V validation")
        pipeline = StableVideoDiffusionPipeline.from_pretrained(
            self.args.model_path,
            vae=self.components.vae,
            text_encoder=self.components.text_encoder,
            tokenizer=self.components.tokenizer,
            transformer=self.components.transformer,
        )
        return pipeline

    def encode_video(self, video: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            latents = self.components.vae.encode(video).latent_dist.sample()
        return latents * self.components.vae.config.scaling_factor

    def encode_text(self, text: str) -> torch.Tensor:
        with torch.no_grad():
            tokens = self.components.tokenizer(text, return_tensors="pt", padding=True, truncation=True).input_ids.to(self.accelerator.device)
            embeddings = self.components.text_encoder(tokens)[0]
        return embeddings

    def compute_loss(self, batch) -> torch.Tensor:
        input_videos = batch["input_videos"]  # [B, C, F_in, H, W]
        target_videos = batch["target_videos"]  # [B, C', F, H', W']
        prompt_embeddings = batch["prompt_embeddings"]  # [B, seq_len, hidden_size]
        
        # Encode input video to latents if not precomputed
        with torch.no_grad():
            input_latents = self.encode_video(input_videos.unsqueeze(0)).squeeze(0) if input_videos.shape[1] == 3 else input_videos
        
        # Forward pass through transformer
        noise = torch.randn_like(target_videos)
        timesteps = torch.randint(0, 1000, (input_latents.shape[0],), device=self.accelerator.device)
        noisy_latents = self.components.transformer.add_noise(target_videos, noise, timesteps)
        predicted_noise = self.components.transformer(noisy_latents, timesteps, encoder_hidden_states=prompt_embeddings, image_latents=input_latents).sample
        
        # Compute MSE loss
        loss = torch.nn.functional.mse_loss(predicted_noise, noise)
        return loss

    def validation_step(self, inputs: Dict[str, Any], pipe: StableVideoDiffusionPipeline) -> List[Image.Image]:
        prompt = inputs["prompt"]
        image = inputs["image"]
        if image is not None:
            image = image.permute(1, 2, 0).cpu().numpy()
            image = Image.fromarray((image * 255).astype('uint8'))
        with torch.no_grad():
            video_frames = pipe(
                image=image,
                prompt=prompt,
                num_frames=self.state.train_frames,
                height=self.state.train_height,
                width=self.state.train_width,
                num_inference_steps=25,
                generator=self.state.generator,
            ).frames[0]
        return video_frames

    def __get_training_dtype(self) -> torch.dtype:
        if self.args.mixed_precision == "no":
            return _DTYPE_MAP["fp32"]
        elif self.args.mixed_precision == "fp16":
            return _DTYPE_MAP["fp16"]
        elif self.args.mixed_precision == "bf16":
            return _DTYPE_MAP["bf16"]
        else:
            raise ValueError(f"Invalid mixed precision: {self.args.mixed_precision}")

    def __move_components_to_device(self, dtype, ignore_list: List[str] = []):
        ignore_list = set(ignore_list)
        components = self.components.model_dump()
        for name, component in components.items():
            if not isinstance(component, type) and hasattr(component, "to"):
                if name not in ignore_list:
                    setattr(self.components, name, component.to(self.accelerator.device, dtype=dtype))

    def __move_components_to_cpu(self, unload_list: List[str] = []):
        unload_list = set(unload_list)
        components = self.components.model_dump()
        for name, component in components.items():
            if not isinstance(component, type) and hasattr(component, "to"):
                if name in unload_list:
                    setattr(self.components, name, component.to("cpu"))

    def __prepare_saving_loading_hooks(self, transformer_lora_config):
        def save_model_hook(models, weights, output_dir):
            if self.accelerator.is_main_process:
                transformer_lora_layers_to_save = None
                for model in models:
                    if isinstance(unwrap_model(self.accelerator, model), type(unwrap_model(self.accelerator, self.components.transformer))):
                        model = unwrap_model(self.accelerator, model)
                        transformer_lora_layers_to_save = get_peft_model_state_dict(model)
                    else:
                        raise ValueError(f"Unexpected save model: {model.__class__}")
                    if weights:
                        weights.pop()
                self.components.pipeline_cls.save_lora_weights(output_dir, transformer_lora_layers=transformer_lora_layers_to_save)

        def load_model_hook(models, input_dir):
            if not self.accelerator.distributed_type == DistributedType.DEEPSPEED:
                while len(models) > 0:
                    model = models.pop()
                    if isinstance(unwrap_model(self.accelerator, model), type(unwrap_model(self.accelerator, self.components.transformer))):
                        transformer_ = unwrap_model(self.accelerator, model)
                    else:
                        raise ValueError(f"Unexpected save model: {unwrap_model(self.accelerator, model).__class__}")
            else:
                transformer_ = unwrap_model(self.accelerator, self.components.transformer).__class__.from_pretrained(self.args.model_path, subfolder="transformer")
                transformer_.add_adapter(transformer_lora_config)
            lora_state_dict = self.components.pipeline_cls.lora_state_dict(input_dir)
            transformer_state_dict = {k.replace("transformer.", ""): v for k, v in lora_state_dict.items() if k.startswith("transformer.")}
            incompatible_keys = set_peft_model_state_dict(transformer_, transformer_state_dict, adapter_name="default")
            if incompatible_keys is not None:
                unexpected_keys = getattr(incompatible_keys, "unexpected_keys", None)
                if unexpected_keys:
                    logger.warning(f"Loading adapter weights from state_dict led to unexpected keys not found in the model: {unexpected_keys}.")

        self.accelerator.register_save_state_pre_hook(save_model_hook)
        self.accelerator.register_load_state_pre_hook(load_model_hook)

    def __maybe_save_checkpoint(self, global_step: int, must_save: bool = False):
        if self.accelerator.distributed_type == DistributedType.DEEPSPEED or self.accelerator.is_main_process:
            if must_save or global_step % self.args.checkpointing_steps == 0:
                save_path = get_intermediate_ckpt_path(checkpointing_limit=self.args.checkpointing_limit, step=global_step, output_dir=self.args.output_dir)
                self.accelerator.save_state(save_path, safe_serialization=True)