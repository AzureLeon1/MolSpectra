#!/bin/bash
gpu_id="0"
job_id="pretrain_baseline"

CUDA_VISIBLE_DEVICES="${gpu_id}" python scripts/train.py \
    --conf examples/ET-QM9-PCQM4MV2-PT.yaml \
    --layernorm-on-vec whitened \
    --job-id "${job_id}" \