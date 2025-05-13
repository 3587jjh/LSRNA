import os
import argparse
import random
import numpy as np
import torch

from diffusers import DDIMScheduler
from pipeline_lsrna_demofusion_sdxl import DemoFusionLSRNASDXLPipeline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prompt', type=str, required=True)
    parser.add_argument('--negative_prompt', type=str)
    parser.add_argument('--height', type=int, default=2048, help='target height')
    parser.add_argument('--width', type=int, default=2048, help='target width')
    parser.add_argument('--seed', type=int)
    parser.add_argument('--lsr_path', type=str, default='lsr/checkpoints/swinir-liif-latent-sdxl.pth')
    parser.add_argument('--rna_min_std', type=float, default=0.0)
    parser.add_argument('--rna_max_std', type=float, default=1.2)
    parser.add_argument('--inversion_depth', type=int, default=30)
    parser.add_argument('--save_dir', type=str, default='results')
    parser.add_argument('--low_vram', action='store_true')
    args = parser.parse_args()

    # load pipeline
    model_ckpt = 'stabilityai/stable-diffusion-xl-base-1.0'
    scheduler = DDIMScheduler.from_pretrained(model_ckpt, subfolder='scheduler')
    pipe = DemoFusionLSRNASDXLPipeline.from_pretrained(model_ckpt, scheduler=scheduler, torch_dtype=torch.float16).to('cuda') 
    pipe.vae.enable_tiling()

    # fix seed
    if args.seed is not None:
        seed = args.seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # generate image (with default setting of DemoFusion)
    images = pipe(
        args.prompt,
        negative_prompt=args.negative_prompt,
        height=args.height, width=args.width, 
        view_batch_size=8, 
        stride_ratio=0.5, # 1-overlap_ratio
        lsr_path=args.lsr_path,
        cosine_scale_1=3,
        cosine_scale_2=1,
        cosine_scale_3=1,
        sigma=0.8,
        rna_min_std=args.rna_min_std,
        rna_max_std=args.rna_max_std,
        inversion_depth=args.inversion_depth,
        low_vram=args.low_vram
    )
    os.makedirs(args.save_dir, exist_ok=True)
    images[0].save(os.path.join(args.save_dir, 'ref.png'))
    images[1].save(os.path.join(args.save_dir, 'trg.png'))


if __name__ == '__main__':
    main()