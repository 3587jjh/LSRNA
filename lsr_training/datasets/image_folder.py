import os
from PIL import Image

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from datasets import register
from utils.utils_io import *


@register('image-folder')
class ImageFolder(Dataset):

    def __init__(self, hr_path, lr_path, first_k=None, repeat=1, scales=[2,3,4]):
        self.repeat = repeat
        self.files = sorted(os.listdir(hr_path))
        if first_k is not None:
            self.files = self.files[:first_k]
        self.hr_path = hr_path
        self.lr_path = lr_path
        self.scales = scales

    def __len__(self):
        return len(self.files) * self.repeat

    def __getitem__(self, idx):
        filename = self.files[idx % len(self.files)]
        hr_path = os.path.join(self.hr_path, filename)

        lr_paths = []
        for scale in self.scales:
            lr_path = os.path.join(self.lr_path, f'X{scale}', filename)
            lr_paths.append(lr_path)
        return hr_path, lr_paths