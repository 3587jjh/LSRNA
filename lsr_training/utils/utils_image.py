import os
from os import path as osp
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torchvision import transforms
IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP']


def get_resolution(img_path):
    img = Image.open(img_path).convert('RGB')
    w,h = img.size
    return (h,w)

def get_image_np(img_path):
    img = Image.open(img_path).convert('RGB')
    return np.array(img)

def get_image_tensor(img_path):
    img = Image.open(img_path).convert('RGB')
    return transforms.ToTensor()(img)


def get_resolutions(folder_path):
    assert os.path.isdir(folder_path), '{:s} is not a valid directory'.format(folder_path)
    resols = []
    for fname in tqdm(sorted(os.listdir(folder_path))):
        if any(fname.endswith(extension) for extension in IMG_EXTENSIONS):
            img_path = os.path.join(folder_path, fname)
            resols.append(get_resolution(img_path))
    return resols

def get_images_np(folder_path):
    assert os.path.isdir(folder_path), '{:s} is not a valid directory'.format(folder_path)
    imgs = []
    for fname in tqdm(sorted(os.listdir(folder_path))):
        if any(fname.endswith(extension) for extension in IMG_EXTENSIONS):
            img_path = os.path.join(folder_path, fname)
            img = get_image_np(img_path)
            imgs.append(img)
    return imgs

def get_images_tensor(folder_path):
    assert os.path.isdir(folder_path), '{:s} is not a valid directory'.format(folder_path)
    imgs = []
    for fname in tqdm(sorted(os.listdir(folder_path))):
        if any(fname.endswith(extension) for extension in IMG_EXTENSIONS):
            img_path = os.path.join(folder_path, fname)
            img = get_image_tensor(img_path)
            imgs.append(img)
    return imgs


def read_img(img_path):
    if img_path.split('.')[-1] == 'npy':
        img = np.load(img_path)
    else:
        img = np.array(Image.open(img_path).convert('RGB')) / 255.
    return img

def random_crop(img, size, return_pos=False):
    assert img.ndim == 3
    if img.shape[0] in [3,4,8] and img.shape[0] < img.shape[1]: # (c,h,w)
        x0 = np.random.randint(0, img.shape[1]-size+1)
        y0 = np.random.randint(0, img.shape[2]-size+1)
        img = img[:, x0: x0+size, y0: y0+size]
    else: # (h,w,c)
        x0 = np.random.randint(0, img.shape[0]-size+1)
        y0 = np.random.randint(0, img.shape[1]-size+1)
        img = img[x0: x0+size, y0: y0+size, :]
    if return_pos:
        return img, (x0, y0)
    else:
        return img

def random_crop_together(hr, lr, lsize, return_pos=False):
    # img: (h,w,c), range independent
    assert lr.shape[0] > lr.shape[-1]
    s = hr.shape[0] // lr.shape[0]
    x0 = np.random.randint(0, lr.shape[0]-lsize+1)
    y0 = np.random.randint(0, lr.shape[1]-lsize+1)
    lr = lr[x0: x0+lsize, y0: y0+lsize, :]
    hr = hr[x0*s: (x0+lsize)*s, y0*s: (y0+lsize)*s, :]
    if return_pos:
        return hr, lr, (x0, y0)
    else:
        return hr, lr

def center_crop(img, size):
    # img: (h,w,3), range independent
    h,w = img.shape[:2]
    cut_h, cut_w = h-size[0], w-size[1]
    
    lh = cut_h // 2
    rh = h - (cut_h - lh)
    lw = cut_w // 2
    rw = w - (cut_w - lw)
    
    img = img[lh:rh, lw:rw, :]
    return img

def tensor2numpy(tensor, rgb_range=1.):
    rgb_coefficient = 255 / rgb_range
    img = tensor.mul(rgb_coefficient).clamp(0, 255).round()
    img = img[0].data if img.ndim==4 else img.data
    img = np.transpose(img.cpu().numpy(), (1, 2, 0)).astype(np.uint8)
    return img