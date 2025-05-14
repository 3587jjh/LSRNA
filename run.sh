#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=0 python main.py \
    --prompt "A well-worn baseball glove and ball sitting on fresh-cut grass." \
    --negative_prompt "blurry, ugly, duplicate, poorly drawn, deformed, mosaic" \
    --height 2048 \
    --width 2048 \
    --seed 0 \
    --lsr_path "lsr/swinir-liif-latent-sdxl.pth" \
    --rna_min_std 0.0 \
    --rna_max_std 1.2 \
    --inversion_depth 30 \
    --save_dir "results" \
    #--low_vram
