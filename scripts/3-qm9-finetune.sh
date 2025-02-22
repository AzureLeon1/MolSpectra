#!/bin/bash
gpu_id="0"
task="mu"

pretrain_model="qm9sp-con_recon-specformer-logx_norm-v3"

job_id="finetune-${pretrain_model}-denoise-${task}"

CUDA_VISIBLE_DEVICES="${gpu_id}" python scripts/train.py \
    --conf examples/ET-QM9-FT.yaml \
    --layernorm-on-vec whitened \
    --job-id "${job_id}" \
    --dataset-arg "${task}" \
    --pretrained-model ./checkpoints/${pretrain_model}.ckpt \
    --contrastive-weight 0 \
