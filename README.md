# PPMI_GEN

## Data preprocessing
```bash
python /workspace/PPMI_GEN/scripts/prepare_dataset.py \
  --ppmi_root /workspace/PPMI \
  --out_root /workspace/mri2datscan/ppmi_mri2datscan \
  --resolution 512 \
  --n_slices 2 \
  --max_delta_days 5000 \
  --max_pairs 50
```
## Train
```bash
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


```