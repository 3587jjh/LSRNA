import warnings
warnings.filterwarnings("ignore")
import os, sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from functools import partial
import argparse
import yaml
import builtins

from utils import *
import datasets
import models
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from diffusers import StableDiffusionXLPipeline


def prepare_training(config, log):
    resume_path = config['resume_path']
    resume = os.path.exists(resume_path)

    if resume:
        sv_file = torch.load(resume_path, map_location=config['map_loc'])
        iter_start = sv_file['iter']+1
        if iter_start <= config['iter_max']//100:
            resume = False
        else:
            log('Model resumed from: {} (prev_iter: {})'.format(resume_path, sv_file['iter']))
            model = models.make(sv_file['model'], load_sd=True).cuda()
            optimizer, lr_scheduler = make_optim_sched(model.parameters(),
                sv_file['optimizer'], sv_file['lr_scheduler'], load_sd=True)

    if not resume:
        if config.get('init_path'):
            log('Model init from: {}'.format(config['init_path']))
            sv_file = torch.load(config['init_path'], map_location=config['map_loc'])    
            model = models.make(sv_file['model'], load_sd=True).cuda()
        else:
            log('Loading new model ...')
            model = models.make(config['model']).cuda()
        optimizer, lr_scheduler = make_optim_sched(model.parameters(),
            config['optimizer'], config['lr_scheduler'])
        iter_start = 1
    log('#params={}'.format(compute_num_params(model, text=True)))

    # load vae
    sd_ckpt = config['sd_ckpt']
    pipeline = StableDiffusionXLPipeline.from_pretrained(sd_ckpt)
    pipeline.enable_vae_tiling()
    vae = pipeline.vae.cuda() # eval mode, float32, i/o range [-1,1]
    return model, optimizer, lr_scheduler, iter_start, vae


def make_train_loader(config):
    spec = config['train_dataset']
    seed = 0 if not config['seed'] else config['seed']
    dataset = datasets.make(spec['dataset'])
    dataset = datasets.make(spec['wrapper'], args={'dataset': dataset})

    assert spec['batch_size'] % config['world_size'] == 0
    batch_size = spec['batch_size'] // config['world_size']
    assert spec['num_workers'] % config['world_size'] == 0
    num_workers = spec['num_workers'] // config['world_size']

    sampler = DistributedSampler(dataset, shuffle=True, seed=seed)
    data_loader = DataLoader(dataset, batch_size=batch_size, drop_last=True, 
        shuffle=False, pin_memory=True, num_workers=num_workers, sampler=sampler)
    return data_loader, sampler


def valid(model, config, vae):
    valid_path = config['valid_path']
    valid_data_name = config['valid_path'].split('/')[-2]
    scale = 2 # fixed
    model.eval()
    
    filenames = sorted(os.listdir(valid_path))
    for filename in tqdm(filenames, leave=True, desc=f'valid (x{scale})'):
        hr_file = os.path.join(valid_path, filename)
        hr = np.array(Image.open(hr_file).convert('RGB')) / 255.
        hr = torch.from_numpy(hr).permute(2,0,1).float().unsqueeze(0).cuda()
          
        with torch.no_grad():
            # crop to divisible size
            H,W = hr.shape[-2:]
            H,W = H//8*8, W//8*8
            hr = hr[:,:H,:W]
            hr = (hr - 0.5) * 2 # normalize to [-1,1]

            hr_latent = vae.encode(hr).latent_dist.mode() * vae.config.scaling_factor
            H,W = hr_latent.shape[-2:]
            H,W = H*scale, W*scale

            coord = make_coord((H,W), flatten=False, device='cuda').unsqueeze(0)
            cell = torch.ones_like(coord)
            cell[:,:,:,0] *= 2/H
            cell[:,:,:,1] *= 2/W

            pred_latent = model(hr_latent, coord, cell)
            pred = vae.decode(pred_latent / vae.config.scaling_factor, return_dict=False)[0]

            # denormalize
            pred = pred / 2 + 0.5 
            hr = hr / 2 + 0.5

        save_dir = os.path.join(config['save_path'], 'valid', valid_data_name, f'X{scale}')
        os.makedirs(save_dir, exist_ok=True)
        filename = filename.split('.')[0] # w/o extension
        Image.fromarray(tensor2numpy(pred)).save(os.path.join(save_dir, f'{filename}_pred.png'))
        Image.fromarray(tensor2numpy(hr)).save(os.path.join(save_dir, f'{filename}_hr.png'))


def main(): 
    # get options
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--launcher', default='pytorch', help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()

    # distributed setting
    init_dist('pytorch')
    rank, world_size = get_dist_info()

    # load logger
    save_path = os.path.join('save', args.config.split('/')[-1][:-len('.yaml')])
    logger = Logger()
    logger.set_save_path(save_path, remove=False)
    if rank > 0: 
        builtins.print = lambda *args, **kwargs: None
        logger.disable()
    log = logger.log

    # load config
    config = load_config(args.config)
    config['world_size'] = world_size
    if config['seed'] is not None:
        set_seed(config['seed'])
    if rank == 0:
        os.makedirs(save_path, exist_ok=True)
        with open(os.path.join(save_path, 'config.yaml'), 'w') as f:
            yaml.dump(config, f, sort_keys=False)
    log('Config loaded: {}'.format(args.config))
    config['rank'] = rank
    config['map_loc'] = f'cuda:{rank}'

    # prepare training
    model, optimizer, lr_scheduler, iter_start, vae = prepare_training(config, log)
    model = nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)
    train_loader, train_sampler = make_train_loader(config)

    if rank == 0:
        assert os.path.exists(config['valid_path'])
        timer = Timer()
        train_loss = Averager()
        t_iter_start = timer.t()

    iter_cur = iter_start
    iter_max = config['iter_max']
    iter_print = config['iter_print']
    iter_val = config['iter_val']
    iter_save = config['iter_save']

    loss_fn = nn.L1Loss()
    while True:
        train_sampler.set_epoch(iter_cur) # instead of epoch
        for batch in train_loader: # process single iteration
            for key, value in batch.items():
                batch[key] = value.cuda()
            model.train()
            optimizer.zero_grad()

            hr, lr = batch['hr'], batch['lr']
            assert hr.shape[1] == lr.shape[1] and hr.shape[1] == 4
            coord, cell = batch['coord'], batch['cell']
            pred = model(lr, coord, cell)
            loss = loss_fn(pred, hr)

            loss.backward()
            optimizer.step()
            lr_scheduler.step()

            if rank == 0:
                train_loss.add(loss.item())
                cond1 = (iter_cur % iter_print == 0)
                cond2 = (iter_cur % iter_save == 0)
                cond3 = (iter_cur % iter_val == 0)

                if cond1 or cond2 or cond3:
                    model_ = model.module if hasattr(model, 'module') else model
                    if cond1 or cond2:
                        # save current model state
                        model_spec = config['model']
                        model_spec['sd'] = model_.state_dict()
                        optimizer_spec = config['optimizer']
                        optimizer_spec['sd'] = optimizer.state_dict()
                        lr_scheduler_spec = config['lr_scheduler']
                        lr_scheduler_spec['sd'] = lr_scheduler.state_dict()
                        sv_file = {
                            'model': model_spec,
                            'optimizer': optimizer_spec,
                            'lr_scheduler': lr_scheduler_spec,
                            'iter': iter_cur
                        }
                        if cond1:
                            log_info = ['iter {}/{}'.format(iter_cur, iter_max)]
                            log_info.append('train: loss={:.4f}'.format(train_loss.item()))
                            log_info.append('lr: {:.4e}'.format(lr_scheduler.get_last_lr()[0]))

                            t = timer.t()
                            prog = (iter_cur - iter_start + 1) / (iter_max - iter_start + 1)
                            t_iter = time_text(t - t_iter_start)
                            t_elapsed, t_all = time_text(t), time_text(t / prog)
                            log_info.append('{} {}/{}'.format(t_iter, t_elapsed, t_all))
                            log(', '.join(log_info))
                            train_loss = Averager()
                            t_iter_start = timer.t()
                            torch.save(sv_file, os.path.join(config['save_path'], 'iter_last.pth'))
                        if cond2:
                            torch.save(sv_file, os.path.join(config['save_path'], 'iter_{}.pth'.format(iter_cur)))
                    if cond3: # validation
                        valid(model_, config, vae=vae)


            if iter_cur == iter_max:
                log('Finish training.')
                return
            iter_cur += 1

if __name__ == '__main__':
    main()
