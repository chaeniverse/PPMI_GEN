import torch
from PIL import Image
from diffusers import StableDiffusionXLControlNetPipeline, ControlNetModel, UniPCMultistepScheduler

BASE = "stabilityai/stable-diffusion-xl-base-1.0"
CTRL = "/workspace/mri2datscan/models/controlnet_sdxl_ppmi"
COND_IMG = "/workspace/mri2datscan/ppmi_mri2datscan/data/train/conditioning/REPLACE_ME.png"

prompt = "brain DaTSCAN SPECT, grayscale"

controlnet = ControlNetModel.from_pretrained(CTRL, torch_dtype=torch.float16)
pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
    BASE,
    controlnet=controlnet,
    torch_dtype=torch.float16,
).to("cuda")

pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
pipe.enable_xformers_memory_efficient_attention()

cond = Image.open(COND_IMG).convert("RGB").resize((512, 512))
out = pipe(prompt=prompt, image=cond, num_inference_steps=30).images[0]
out.save("/workspace/mri2datscan/out.png")
print("saved: /workspace/mri2datscan/out.png")

