#!/bin/bash
gpu_id="0"
job_id="md17_pretrain_baseline"

CUDA_VISIBLE_DEVICES="${gpu_id}" python scripts/train.py \
    --conf examples/ET-MD17-PCQM4MV2-PT.yaml \
    --layernorm-on-vec whitened \
    --job-id "${job_id}" \