# PPMI_GEN

## Data preprocessing
```bash
python /workspace/PPMI_GEN/scripts/prepare_dataset.py \
  --ppmi_root /workspace/PPMI \
  --out_root /workspace/mri2datscan/ppmi_mri2datscan \
  --resolution 512 \
  --n_slices 7 \
  --max_delta_days 365 \
  --max_pairs 50 \
  --dat_crop 160 \
  --dat_crop_mode hotspot


```
## Train
```bash
export TRAIN_DIR=/workspace/mri2datscan/ppmi_mri2datscan/data/train
cd /workspace/PPMI_GEN/diffusers/examples/controlnet

accelerate launch \
  --num_processes 1 \
  --mixed_precision fp16 \
  train_controlnet_sdxl.py \
  --pretrained_model_name_or_path stabilityai/stable-diffusion-xl-base-1.0 \
  --variant fp16 \
  --mixed_precision fp16 \
  --train_data_dir "$TRAIN_DIR" \
  --image_column image \
  --conditioning_image_column conditioning_image \
  --caption_column text \
  --resolution 512 \
  --learning_rate 1e-5 \
  --train_batch_size 1 \
  --gradient_accumulation_steps 4 \
  --max_train_steps 20000 \
  --checkpointing_steps 1000 \
  --validation_prompt "brain DaTSCAN SPECT, grayscale" \
  --validation_image "$TRAIN_DIR/conditioning/$(ls "$TRAIN_DIR/conditioning" | head -n 1)" \
  --num_validation_images 2 \
  --gradient_checkpointing \
  --use_8bit_adam \
  --output_dir /workspace/mri2datscan/models/controlnet_sdxl_ppmi

```

## Inference
```bash
import torch
from diffusers import ControlNetModel, StableDiffusionXLControlNetPipeline
from PIL import Image

ckpt = "/workspace/mri2datscan/models/controlnet_sdxl_ppmi/checkpoint-1000"

# 1) controlnet을 fp16으로 로드 (핵심)
cn = ControlNetModel.from_pretrained(
    ckpt,
    subfolder="controlnet",
    torch_dtype=torch.float16,
)

# 2) SDXL base도 fp16 variant로 로드
pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    controlnet=cn,
    torch_dtype=torch.float16,
    variant="fp16",
    use_safetensors=True,
).to("cuda")

# 3) 혹시 남아있는 float32 강제 캐스팅 (안전장치)
pipe.controlnet.to(dtype=torch.float16)
pipe.unet.to(dtype=torch.float16)
pipe.vae.to(dtype=torch.float16)
pipe.text_encoder.to(dtype=torch.float16)
pipe.text_encoder_2.to(dtype=torch.float16)

cond = Image.open(
    "/workspace/mri2datscan/ppmi_mri2datscan/data/train/conditioning/3107_dt2011-07-20_t12011-04-13_z031_00.png"
).convert("RGB")

prompt = "brain DaTSCAN SPECT, grayscale"

with torch.autocast("cuda", dtype=torch.float16):
    img = pipe(prompt=prompt, image=cond, num_inference_steps=30).images[0]

out_path = "/workspace/mri2datscan/models/controlnet_sdxl_ppmi/sample_from_ckpt1000.png"
img.save(out_path)
print("saved to:", out_path)


```