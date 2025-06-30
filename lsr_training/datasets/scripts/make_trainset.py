import random
import os
import pandas as pd
import requests
from PIL import Image
from io import BytesIO
from tqdm import tqdm
import argparse
import pickle

import numpy as np
import torch
import torchvision.transforms as transforms
from diffusers import StableDiffusionXLPipeline

import sys
sys.path.append('../..')
import core

#random.seed(0)
#np.random.seed(0)
#torch.manual_seed(0)
#torch.cuda.manual_seed_all(0)

parser = argparse.ArgumentParser(description='OpenImages downloader')
parser.add_argument('--max_sample', type=int, default=1560000) # per part 
parser.add_argument('--part', type=str, default='1/1') 
args = parser.parse_args()

down_scales = [2,3,4] # fixed
base_dir = '/workspace/datasets/train/OpenImages' # fixed
count = 0

annotation_path = f'{base_dir}/image_ids_and_rotation.csv' # metadata of OpenImages 
print('loading annotation file...')
a,b=map(int, args.part.split('/'))
urls = list(pd.read_csv(annotation_path)['OriginalURL'])[(a-1)::b]

processed_info = {}
processed_info_path = f'{base_dir}/process_info_{a}_{b}.pkl'
if os.path.exists(processed_info_path):
    with open(f'{base_dir}/process_info_{a}_{b}.pkl', 'rb') as f:
        processed_info = pickle.load(f)

def get_image(url):
    global count, processed_info
    session = requests.Session()
    try:
        img_name = url.split('/')[-1].split('?')[0]
        if img_name[-4:].lower() not in ['.jpg', 'jpeg']:
            return None, None
        assert img_name.count('.') == 1
        img_name = img_name.split('.')[0] # w/o extension

        key = f'{base_dir}/HR/{img_name}_s000.jpg'
        if key in processed_info:
            count += processed_info[key]
            print(f'[skip] files already exists for {img_name} | count: {count}')
            return None, None

        response = session.get(url, timeout=2)
        response.raise_for_status() 
        img = Image.open(BytesIO(response.content))

        width, height = img.size
        if height >= 1440 and width >= 1440 and img.mode == 'RGB':
            return img, img_name
        return None, None

    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
        return None, None
    except Exception as e:
        print(f"Other error occurred: {e}")
        return None, None
    finally:
        session.close()

os.makedirs(f'{base_dir}/HR', exist_ok=True)
os.makedirs(f'{base_dir}/HR_sdxl_latent', exist_ok=True)
for down_scale in down_scales:
    os.makedirs(f'{base_dir}/LR/X{down_scale}', exist_ok=True)
    os.makedirs(f'{base_dir}/LR_sdxl_latent/X{down_scale}', exist_ok=True)

sd_ckpt = 'stabilityai/stable-diffusion-xl-base-1.0'
pipeline = StableDiffusionXLPipeline.from_pretrained(sd_ckpt)
vae = pipeline.vae.cuda() # eval mode, float32, i/o range [-1,1]

for url in urls:
    if count >= args.max_sample: 
        print(f'count ({count}) reached the max_sample={args.max_sample}')
        break
    img, base_name = get_image(url)
    if img is None: continue

    # found new HR image
    crop_size = random.randint(1056,1440)//96*96
    step = crop_size
    w,h = img.size

    h_space = np.arange(0, h-crop_size+1, step)
    if h > h_space[-1] + crop_size:
        h_space = np.append(h_space, h-crop_size)
    w_space = np.arange(0, w-crop_size+1, step)
    if w > w_space[-1] + crop_size:
        w_space = np.append(w_space, w-crop_size)

    hrs = []
    for x in h_space:
        for y in w_space:
            hr = img.crop((y, x, y+crop_size, x+crop_size))
            hrs.append(hr)
    hrs = hrs[::-1]
    
    for i, hr in enumerate(tqdm(hrs)):
        index = len(hrs)-i-1
        name = f'{base_name}_s{index:03d}' # w/o extension
        hr = transforms.ToTensor()(hr).unsqueeze(0).cuda() # (1,3,csz,csz), range [0,1]
        with torch.no_grad():
            hr_latent = vae.encode((hr-0.5)*2).latent_dist.mode() * vae.config.scaling_factor
        
        # bicubic degradation & conv_latent
        for down_scale in down_scales:
            lr = core.imresize(hr, sizes=(crop_size//down_scale, crop_size//down_scale))
            lr = (lr*255).clip(0,255).to(torch.uint8).float() / 255 # discretized [0,1]
            transforms.ToPILImage()(lr.squeeze(0)).save(f'{base_dir}/LR/X{down_scale}/{name}.jpg')

            with torch.no_grad():
                lr_latent = vae.encode((lr-0.5)*2).latent_dist.mode() * vae.config.scaling_factor
            np.save(f'{base_dir}/LR_sdxl_latent/X{down_scale}/{name}.npy', 
                lr_latent.squeeze(0).permute(1,2,0).detach().cpu().numpy())
                
        np.save(f'{base_dir}/HR_sdxl_latent/{name}.npy', 
            hr_latent.squeeze(0).permute(1,2,0).detach().cpu().numpy())
        transforms.ToPILImage()(hr.squeeze(0)).save(f'{base_dir}/HR/{name}.jpg')

        if index == 0:
            key = f'{base_dir}/HR/{name}.jpg'
            assert key not in processed_info
            processed_info[key] = len(hrs)
            with open(processed_info_path, 'wb') as f:
                pickle.dump(processed_info, f)
            count += len(hrs)
            print(f'count: {count} / {args.max_sample} | succesfully processed {base_name}')
