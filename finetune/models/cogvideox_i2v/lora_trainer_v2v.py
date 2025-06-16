from typing import Any, Dict, List, Tuple
import torch
from diffusers import (
    AutoencoderKLCogVideoX,
    CogVideoXDPMScheduler,
    CogVideoXImageToVideoPipeline,
    CogVideoXVideoToVideoPipeline,
    CogVideoXTransformer3DModel,
)
from diffusers.models.embeddings import get_3d_rotary_pos_embed
from PIL import Image
import numpy as np
from transformers import AutoTokenizer, T5EncoderModel
from typing_extensions import override
from finetune.schemas import Components
from finetune.trainer import Trainer
from finetune.utils import unwrap_model
from ..utils import register

class CogVideoXV2VLoraTrainer(Trainer):
    UNLOAD_LIST = ["text_encoder"]

    @override
    def load_components(self) -> Dict[str, Any]:
        components = Components()
        model_path = str(self.args.model_path)

        # Allow switching between I2V and V2V pipelines
        components.pipeline_cls = (
            CogVideoXVideoToVideoPipeline if hasattr(self, "use_v2v_pipeline") and self.use_v2v_pipeline
            else CogVideoXImageToVideoPipeline
        )

        components.tokenizer = AutoTokenizer.from_pretrained(model_path, subfolder="tokenizer")
        components.text_encoder = T5EncoderModel.from_pretrained(model_path, subfolder="text_encoder")
        components.transformer = CogVideoXTransformer3DModel.from_pretrained(model_path, subfolder="transformer")
        components.vae = AutoencoderKLCogVideoX.from_pretrained(model_path, subfolder="vae")
        components.scheduler = CogVideoXDPMScheduler.from_pretrained(model_path, subfolder="scheduler")

        return components

    @override
    def initialize_pipeline(self) -> CogVideoXImageToVideoPipeline | CogVideoXVideoToVideoPipeline:
        pipe_cls = self.components.pipeline_cls
        pipe = pipe_cls(
            tokenizer=self.components.tokenizer,
            text_encoder=self.components.text_encoder,
            vae=self.components.vae,
            transformer=unwrap_model(self.accelerator, self.components.transformer),
            scheduler=self.components.scheduler,
        )
        return pipe

    @override
    def encode_video(self, video: torch.Tensor) -> torch.Tensor:
        """Encode a video into latent space using the VAE."""
        vae = self.components.vae
        video = video.to(vae.device, dtype=vae.dtype)  # [B, C, F, H, W]
        latent_dist = vae.encode(video).latent_dist
        latent = latent_dist.sample() * vae.config.scaling_factor  # [B, C, F, H_latent, W_latent]
        return latent

    @override
    def encode_text(self, prompt: str) -> torch.Tensor:
        """Encode a text prompt using the tokenizer and text encoder."""
        prompt_token_ids = self.components.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.state.transformer_config.max_text_seq_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        ).input_ids
        return self.components.text_encoder(prompt_token_ids.to(self.accelerator.device))[0]

    @override
    def collate_fn(self, samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Collate input videos, target videos, and prompts into a batch."""
        ret = {"encoded_videos": [], "prompt_embedding": [], "input_videos": []}

        for sample in samples:
            target_video = sample["target_video"]  # [F_out, C, H, W]
            encoded_video = self.encode_video(target_video.unsqueeze(0)).squeeze(0)  # [C, F_out, H_latent, W_latent]
            prompt_embedding = self.encode_text(sample["prompt"])
            input_video = sample["input_video"]  # [F_in, C, H, W]

            ret["encoded_videos"].append(encoded_video)
            ret["prompt_embedding"].append(prompt_embedding)
            ret["input_videos"].append(input_video)

        ret["encoded_videos"] = torch.stack(ret["encoded_videos"])    # [B, C, F_out, H_latent, W_latent]
        ret["prompt_embedding"] = torch.stack(ret["prompt_embedding"]) # [B, seq_len, hidden_size]
        ret["input_videos"] = torch.stack(ret["input_videos"])        # [B, C, F_in, H, W]

        return ret

    @override
    def compute_loss(self, batch) -> torch.Tensor:
        """Compute the loss for V2V training."""
        prompt_embedding = batch["prompt_embedding"]  # [B, seq_len, hidden_size]
        latent = batch["encoded_videos"]              # [B, C, F_out, H_latent, W_latent]
        input_videos = batch["input_videos"]          # [B, C, F_in, H, W]

        # Handle patch size alignment for output latent
        patch_size_t = self.state.transformer_config.patch_size_t
        if patch_size_t is not None:
            ncopy = latent.shape[2] % patch_size_t
            if ncopy != 0:
                first_frame = latent[:, :, :1, :, :]
                latent = torch.cat([first_frame.repeat(1, 1, ncopy, 1, 1), latent], dim=2)
            assert latent.shape[2] % patch_size_t == 0

        batch_size, num_channels, num_frames, height, width = latent.shape

        # Prepare prompt embeddings
        prompt_embedding = prompt_embedding.view(batch_size, -1, prompt_embedding.shape[-1]).to(dtype=latent.dtype)

        # Encode input videos to latent space (before adding noise)
        input_latent_dist = self.components.vae.encode(
            input_videos.to(dtype=self.components.vae.dtype)
        ).latent_dist
        input_latents = input_latent_dist.sample() * self.components.vae.config.scaling_factor  # [B, C, F_in, H_latent, W_latent]

        # Add noise to input latents in latent space
        video_noise_sigma = torch.normal(
            mean=-3.0, std=0.5, size=(1,), device=self.accelerator.device
        ).exp().to(dtype=input_latents.dtype)
        noisy_input_latents = (
            input_latents + torch.randn_like(input_latents) * video_noise_sigma[:, None, None, None, None]
        )

        # Permute to [B, F_in, C, H_latent, W_latent]
        noisy_input_latents = noisy_input_latents.permute(0, 2, 1, 3, 4)

        # Adjust input latents to match F_out
        F_out = latent.shape[2]
        if noisy_input_latents.shape[1] < F_out:
            padding_shape = (batch_size, F_out - noisy_input_latents.shape[1], *noisy_input_latents.shape[2:])
            latent_padding = noisy_input_latents[:, -1:, :, :, :].repeat(1, padding_shape[1], 1, 1, 1)
            input_latents_padded = torch.cat([noisy_input_latents, latent_padding], dim=1)  # [B, F_out, C, H_latent, W_latent]
        elif noisy_input_latents.shape[1] > F_out:
            input_latents_padded = noisy_input_latents[:, :F_out, :, :, :]
        else:
            input_latents_padded = noisy_input_latents

        # Prepare target latent
        latent = latent.permute(0, 2, 1, 3, 4)  # [B, F_out, C, H_latent, W_latent]

        # Add noise to target latent
        timesteps = torch.randint(
            0,
            self.components.scheduler.config.num_train_timesteps,
            (batch_size,),
            device=self.accelerator.device,
        ).long()
        noise = torch.randn_like(latent)
        latent_noisy = self.components.scheduler.add_noise(latent, noise, timesteps)

        # Concatenate noisy latents
        latent_img_noisy = torch.cat([latent_noisy, input_latents_padded], dim=2)  # [B, F_out, 2*C, H_latent, W_latent]

        # Prepare rotary embeddings
        vae_scale_factor_spatial = 2 ** (len(self.components.vae.config.block_out_channels) - 1)
        transformer_config = self.state.transformer_config
        rotary_emb = (
            self.prepare_rotary_positional_embeddings(
                height=height * vae_scale_factor_spatial,
                width=width * vae_scale_factor_spatial,
                num_frames=num_frames,
                transformer_config=transformer_config,
                vae_scale_factor_spatial=vae_scale_factor_spatial,
                device=self.accelerator.device,
            )
            if transformer_config.use_rotary_positional_embeddings
            else None
        )

        # Predict noise
        ofs_emb = (
            None
            if self.state.transformer_config.ofs_embed_dim is None
            else latent.new_full((1,), fill_value=2.0)
        )
        predicted_noise = self.components.transformer(
            hidden_states=latent_img_noisy,
            encoder_hidden_states=prompt_embedding,
            timestep=timesteps,
            ofs=ofs_emb,
            image_rotary_emb=rotary_emb,
            return_dict=False,
        )[0]

        # Compute loss
        latent_pred = self.components.scheduler.get_velocity(predicted_noise, latent_noisy, timesteps)
        alphas_cumprod = self.components.scheduler.alphas_cumprod[timesteps]
        weights = 1 / (1 - alphas_cumprod)
        while len(weights.shape) < len(latent_pred.shape):
            weights = weights.unsqueeze(-1)
        loss = torch.mean((weights * (latent_pred - latent) ** 2).reshape(batch_size, -1), dim=1)
        return loss.mean()

    @override
    def validation_step(
        self, eval_data: Dict[str, Any], pipe: CogVideoXImageToVideoPipeline | CogVideoXVideoToVideoPipeline
    ) -> List[Tuple[str, List[Image.Image]]]:
        """Perform validation using the first frame (I2V) or full video (V2V) depending on pipeline."""
        prompt, video = eval_data["prompt"], eval_data["video"]
        if video is None:
            raise ValueError("Input video is required for V2V validation.")

        if isinstance(pipe, CogVideoXVideoToVideoPipeline):
            # Use full video if V2V pipeline is available
            video_tensor = video.unsqueeze(0).to(self.accelerator.device)  # [1, F, C, H, W]
            video_generate = pipe(
                num_frames=self.state.train_frames,
                height=self.state.train_height,
                width=self.state.train_width,
                prompt=prompt,
                video=video_tensor,
                generator=self.state.generator,
            ).frames[0]
        else:
            # Fallback to I2V pipeline using first frame
            first_frame = video[0].permute(1, 2, 0).cpu().numpy()
            first_frame = (first_frame * 255).clip(0, 255).astype(np.uint8)
            first_frame_pil = Image.fromarray(first_frame)
            video_generate = pipe(
                num_frames=self.state.train_frames,
                height=self.state.train_height,
                width=self.state.train_width,
                prompt=prompt,
                image=first_frame_pil,
                generator=self.state.generator,
            ).frames[0]

        return [("video", video_generate)]

    def prepare_rotary_positional_embeddings(
        self, height: int, width: int, num_frames: int, transformer_config: Dict,
        vae_scale_factor_spatial: int, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Prepare rotary positional embeddings for the transformer."""
        grid_height = height // (vae_scale_factor_spatial * transformer_config.patch_size)
        grid_width = width // (vae_scale_factor_spatial * transformer_config.patch_size)

        if transformer_config.patch_size_t is None:
            base_num_frames = num_frames
        else:
            base_num_frames = (num_frames + transformer_config.patch_size_t - 1) // transformer_config.patch_size_t

        freqs_cos, freqs_sin = get_3d_rotary_pos_embed(
            embed_dim=transformer_config.attention_head_dim,
            crops_coords=None,
            grid_size=(grid_height, grid_width),
            temporal_size=base_num_frames,
            grid_type="slice",
            max_size=(grid_height, grid_width),
            device=device,
        )
        return freqs_cos, freqs_sin

register("cogvideox-v2v", "lora", CogVideoXV2VLoraTrainer)