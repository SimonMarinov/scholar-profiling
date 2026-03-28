#!/bin/bash
# Run inference on a local machine (CUDA, Apple-Silicon MPS, or CPU).
# Optionally set MODEL_PATH to a local directory; defaults to the HuggingFace Hub.
MODEL_PATH="${MODEL_PATH:-microsoft/deberta-v3-large}"

python predict.py \
    --task scholar-xl \
    --cnn_dim 32 \
    --biaffine_size 100 \
    --chunks_size 128 \
    --logit_drop 0.1 \
    --cnn_depth 2 \
    --model_path "$MODEL_PATH"
