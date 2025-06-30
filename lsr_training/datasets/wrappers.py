import numpy as np
from torch.utils.data import Dataset
from datasets import register
from utils import *


@register('sr-explicit-paired')
class SRExplicitPaired(Dataset):

    def __init__(self, dataset, inp_size, augment=[], sample_size=None, num_channels=None):
        self.dataset = dataset
        self.inp_size = inp_size
        self.augment = augment
        self.sample_size = inp_size if sample_size is None else sample_size
        self.num_channels = num_channels

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        hr_path, lr_paths = self.dataset[idx]
        lr_path = lr_paths[np.random.randint(len(lr_paths))]

        # img: (H,W,C), numpy, range [-3,3] or [0,1]
        hr, lr = read_img(hr_path), read_img(lr_path)
        if self.num_channels:
            assert hr.shape[-1] == lr.shape[-1] == self.num_channels
        hr, lr = random_crop_together(hr, lr, self.inp_size)

        # augmentation
        hflip = (np.random.random() < 0.5) if 'hflip' in self.augment else False
        vflip = (np.random.random() < 0.5) if 'vflip' in self.augment else False
        dflip = (np.random.random() < 0.5) if 'dflip' in self.augment else False

        def base_augment(img):
            if hflip:
                img = img[::-1, :, :]
            if vflip:
                img = img[:, ::-1, :]
            if dflip:
                img = np.transpose(img, (1, 0, 2))
            return img.copy()
        hr = torch.from_numpy(base_augment(hr)).permute(2,0,1).float() # (C,H,W)
        lr = torch.from_numpy(base_augment(lr)).permute(2,0,1).float() # (C,h,w)

        coord = make_coord(hr.shape[-2:], flatten=False) # (H,W,2)
        cell = torch.ones_like(coord) # (H,W,2)
        cell[:,:,0] *= 2 / hr.shape[-2]
        cell[:,:,1] *= 2 / hr.shape[-1]

        P = self.sample_size
        hr, pos = random_crop(hr, P, return_pos=True) # (C,P,P)
        coord = coord[pos[0]:pos[0]+P, pos[1]:pos[1]+P] # (P,P,2)
        cell = cell[pos[0]:pos[0]+P, pos[1]:pos[1]+P] # (P,P,2)

        return {
            'lr': lr,
            'coord': coord,
            'cell': cell,
            'hr': hr
        }
