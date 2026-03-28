#!/bin/bash
# Train on a local machine (CUDA, Apple-Silicon MPS, or CPU).
# Optionally set MODEL_PATH to a local directory; defaults to the HuggingFace Hub.
MODEL_PATH="${MODEL_PATH:-microsoft/deberta-v3-large}"

python train.py \
    --task scholar-xl \
    --n_epochs 30 \
    --lr 7e-4 \
    --cnn_dim 32 \
    --biaffine_size 100 \
    --chunks_size 128 \
    --batch_size 16 \
    --logit_drop 0.1 \
    --cnn_depth 2 \
    --model_path "$MODEL_PATH"
