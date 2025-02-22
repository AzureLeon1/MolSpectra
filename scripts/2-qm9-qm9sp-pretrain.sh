#!/bin/bash

gpu_id="1"
input_data_norm_type="log10"
uv_model="SpecFormer"
model_name="MultiSpec-con_recon-${uv_model}"
job_id="pretrain-${model_name}-${input_data_norm_type}-no_lr_reduce"

config="examples/ET-QM9-QM9SP-PT.yaml"

CUDA_VISIBLE_DEVICES="${gpu_id}" python scripts/train.py \
    --conf  "${config}"\
    --layernorm-on-vec whitened \
    --job-id "${job_id}" \
    --pretrained-model ./checkpoints/denoised-pcqm4mv2-v2.ckpt \
    --denoising-weight 1.0 \
    --uv-model "${uv_model}" \
    --input-data-norm-type "${input_data_norm_type}" \
    --contrastive-weight 1.0 \
    --reconstruct-weight 1.0 \
    --patch-len 20 50 50 \
    --stride 10 25 25 \
    --mask-ratios 0.1 0.1 0.1 \
