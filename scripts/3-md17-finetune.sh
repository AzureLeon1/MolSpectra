#!/bin/bash
gpu_id="0"
task="aspirin" # aspirin benzene ethanol malonaldehyde naphthalene salicylic_acid toluene uracil

pretrain_model="denoised-pcqm4mv2-md17_setting2"

CUDA_VISIBLE_DEVICES="${gpu_id}" python scripts/train.py \
    --conf examples/ET-MD17-FT.yaml \
    --layernorm-on-vec whitened \
    --job-id "md17_finetunet-${pretrain_model}-${task}" \
    --dataset-arg "${task}" \
    --pretrained-model "checkpoints/${pretrain_model}.ckpt"