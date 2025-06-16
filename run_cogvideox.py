import torch
from diffusers import CogVideoXPipeline
from diffusers.utils import export_to_video

prompt = "A panda, dressed in a small, green T-shirt and a tiny hat, sits on a iron stool in a serene oak forest. The panda's fluffy paws strum a miniature acoustic guitar, producing soft, melodic tunes. Nearby, a few other pandas gather, watching curiously and some clapping in rhythm. Sunlight filters through the tall bamboo, casting a gentle glow on the scene. The panda's face is expressive, showing concentration and joy as it plays. The background includes a small, flowing stream and vibrant green foliage, enhancing the peaceful and magical atmosphere of this unique musical performance."

pipe = CogVideoXPipeline.from_pretrained(
    "THUDM/CogVideoX1.5-5B",
    torch_dtype=torch.bfloat16,
    cache_dir="/home/kmwuab/CogVideo/cogvideox_models"  
)

pipe.enable_sequential_cpu_offload()  
pipe.vae.enable_tiling()       
pipe.vae.enable_slicing()  

# Generate the video
video = pipe(
    prompt=prompt,
    num_videos_per_prompt=1,
    num_inference_steps=50,
    num_frames=81,
    guidance_scale=6,
    generator=torch.Generator(device="cuda").manual_seed(42),  # Use "cpu" if no GPU
).frames[0]

export_to_video(video, "output_new.mp4", fps=8)