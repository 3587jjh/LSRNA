import torch
import torch.nn as nn
import torch.nn.functional as F

from .models import register
from . models import make as make_model

def make_coord(shape, ranges=None, flatten=True, device='cpu'):
    # Make coordinates at grid centers.
    coord_seqs = []
    for i, n in enumerate(shape):
        if ranges is None:
            v0, v1 = -1, 1
        else:
            v0, v1 = ranges[i]
        r = (v1 - v0) / (2 * n)
        seq = v0 + r + (2 * r) * torch.arange(n, device=device).float()
        coord_seqs.append(seq)
    ret = torch.stack(torch.meshgrid(*coord_seqs), dim=-1)
    if flatten:
        ret = ret.view(-1, ret.shape[-1])
    return ret

@register('liif')
class LIIF(nn.Module):

    def __init__(self, encoder_spec, imnet_spec, feat_unfold=True, local_ensemble=True):
        super().__init__()
        self.local_ensemble = local_ensemble
        self.feat_unfold = feat_unfold
        self.encoder = make_model(encoder_spec)

        imnet_in_dim = self.encoder.out_dim
        if self.feat_unfold:
            imnet_in_dim *= 9
        imnet_in_dim += 4 # attach coord, cell
        self.imnet = make_model(imnet_spec, args={'in_dim': imnet_in_dim})
        
    def gen_feat(self, inp):
        self.inp = inp
        feat = self.encoder(inp)
        if self.feat_unfold:
            feat = F.unfold(feat, 3, padding=1).view(
                feat.shape[0], feat.shape[1] * 9, feat.shape[2], feat.shape[3])
        self.feat = feat
        self.feat_coord = make_coord(feat.shape[-2:], flatten=False).cuda() \
            .permute(2, 0, 1) \
            .unsqueeze(0).expand(feat.shape[0], 2, *feat.shape[-2:])
        
    def query_rgb(self, coord, cell):
        # coord, cell: (b,h,w,c)
        feat = self.feat
        feat_coord = self.feat_coord
        if self.local_ensemble:
            vx_lst = [-1, 1]
            vy_lst = [-1, 1]
            eps_shift = 1e-6
        else:
            vx_lst, vy_lst, eps_shift = [0], [0], 0

        # field radius (global: [-1, 1])
        rx = 2 / feat.shape[-2] / 2
        ry = 2 / feat.shape[-1] / 2

        preds = []
        areas = []
        for vx in vx_lst:
            for vy in vy_lst:
                coord_ = coord.clone()
                coord_[:, :, :, 0] += vx * rx + eps_shift
                coord_[:, :, :, 1] += vy * ry + eps_shift
                coord_.clamp_(-1 + 1e-6, 1 - 1e-6)

                q_feat = F.grid_sample(feat, coord_.flip(-1),
                    mode='nearest', align_corners=False).permute(0, 2, 3, 1) # (b,h,w,c)
                q_coord = F.grid_sample(feat_coord, coord_.flip(-1),
                    mode='nearest', align_corners=False).permute(0, 2, 3, 1)

                rel_coord = coord - q_coord
                rel_coord[:, :, :, 0] *= feat.shape[-2]
                rel_coord[:, :, :, 1] *= feat.shape[-1]
                inp = torch.cat([q_feat, rel_coord], dim=-1)

                rel_cell = cell.clone()
                rel_cell[:, :, :, 0] *= feat.shape[-2]
                rel_cell[:, :, :, 1] *= feat.shape[-1]
                inp = torch.cat([inp, rel_cell], dim=-1) # (b,h,w,c)

                pred = self.imnet(inp.contiguous())
                preds.append(pred)

                area = torch.abs(rel_coord[:, :, :, 0] * rel_coord[:, :, :, 1]) # (b,h,w)
                areas.append(area + 1e-9)

        tot_area = torch.stack(areas).sum(dim=0) # (b,h,w)
        if self.local_ensemble:
            t = areas[0]; areas[0] = areas[3]; areas[3] = t
            t = areas[1]; areas[1] = areas[2]; areas[2] = t
        ret = 0
        for pred, area in zip(preds, areas):
            ret = ret + pred * (area / tot_area).unsqueeze(-1)
        ret = ret.permute(0,3,1,2)
        if ret.shape[1] != self.inp.shape[1]:
            ret[:,:-1,:,:] += F.grid_sample(self.inp, coord.flip(-1), mode='bicubic',\
                padding_mode='border', align_corners=False)
        else:
            ret += F.grid_sample(self.inp, coord.flip(-1), mode='bicubic',\
                padding_mode='border', align_corners=False)
        return ret

    def forward(self, inp, coord, cell):
        self.gen_feat(inp)
        #return self.query_rgb(coord, cell)
        H,W = coord.shape[1:3]
        n = H*W
        coord = coord.view(1,1,n,2)
        cell = cell.view(1,1,n,2)

        ql = 0
        preds = None
        while ql < n:
            qr = min(ql + 512*512, n)
            pred = self.query_rgb(coord[:,:,ql:qr,:], cell[:,:,ql:qr,:])
            preds = pred if preds is None else torch.cat([preds, pred], dim=-1)
            ql = qr
        preds = preds.view(1,-1,H,W)
        return preds