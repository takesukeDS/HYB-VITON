import json
import argparse
import random

import numpy as np
import torch.nn.functional as F
import torch
from torch import nn

def save_args(args, to_path):
    with open(to_path, "w") as f:
        json.dump(args.__dict__, f, indent=2)
def load_args(from_path, is_test=True):
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    with open(from_path, "r") as f:
        args.__dict__ = json.load(f)
    args.is_test = is_test
    if "E_name" not in args.__dict__.keys():
        args.E_name = "basic"
    return args   
def tensor2img(x, round=False):
    '''
    x : [BS x c x H x W] or [c x H x W]
    '''
    if x.ndim == 3:
        x = x.unsqueeze(0)
    BS, C, H, W = x.shape
    x = x.permute(0,2,3,1).reshape(-1, W, C).detach().cpu().numpy()
    # x = (x+1)/2
    # x = np.clip(x, 0, 1)
    x = np.clip(x, -1, 1)
    x = (x+1)/2
    adjust = 0.5 if round else 0
    x = np.uint8(x*255.0 + adjust)
    if x.shape[-1] == 1:  # gray sclae
        x = np.concatenate([x,x,x], axis=-1)
    return x
def resize_mask(m, shape):
    m = F.interpolate(m, shape)
    m[m > 0.5] = 1
    m[m < 0.5] = 0
    return m

def remove_overlap(seg_out, warped_cm, inference=False):
    assert len(warped_cm.shape) == 4
    overlapped_region = (torch.cat([seg_out[:, 1:3, :, :], seg_out[:, 5:, :, :]], dim=1)).sum(dim=1, keepdim=True)
    if inference:
        overlapped_region = (overlapped_region > 0.5).float()
    warped_cm = warped_cm - overlapped_region * warped_cm
    return warped_cm

def bilateral_filter(image, kernel_size, sigma_d, sigma_r):
    """Bilateral filter implementation.
        Args:
            image: input float tensor with shape [bsz, ch, height, width]
            kernel_size: int. we assume it is odd.
            sigma_d: float. standard deviation for distance.
            sigma_r: float or tensor. standard deviation for range.
    """
    padding = (kernel_size - 1) // 2
    # distance
    bsz, ch, height, width = image.shape
    if isinstance(sigma_r, float):
        sigma_r = torch.tensor([sigma_r]).expand(bsz)
    sigma_r = sigma_r.to(image.device)
    height_pad = height + 2 * padding
    width_pad = width + 2 * padding
    # gaussian on spacial distance
    grid_x, grid_y = torch.meshgrid(torch.arange(width_pad), torch.arange(height_pad),
                                    indexing='xy')
    grid_x = grid_x.float().to(image.device)
    grid_y = grid_y.float().to(image.device)
    unfold_grid = nn.Unfold(kernel_size=kernel_size)
    grid_x_unfolded = unfold_grid(grid_x[None, None])
    grid_y_unfolded = unfold_grid(grid_y[None, None])
    grid_x_unfolded = grid_x_unfolded.transpose(1, 2).reshape(height * width, 1, kernel_size ** 2)
    grid_y_unfolded = grid_y_unfolded.transpose(1, 2).reshape(height * width, 1, kernel_size ** 2)
    center_index = kernel_size ** 2 // 2
    diff_x_unfolded = grid_x_unfolded - grid_x_unfolded[:, :, center_index][:, :, None]
    diff_y_unfolded = grid_y_unfolded - grid_y_unfolded[:, :, center_index][:, :, None]
    dist_unfolded = diff_x_unfolded ** 2 + diff_y_unfolded ** 2
    gaussian_dist = torch.exp(-dist_unfolded / (2 * sigma_d ** 2))

    # gaussian on range
    unfold = nn.Unfold(kernel_size=kernel_size, padding=padding)
    image_unfolded = unfold(image)
    image_unfolded = image_unfolded.transpose(1, 2).reshape(bsz, height * width, ch, kernel_size ** 2)
    center_value = image_unfolded[:, :, :, center_index]
    diff_value = image_unfolded - center_value[:, :, :, None]
    dist_value = diff_value ** 2
    gaussian_value = torch.exp(-dist_value / (2 * sigma_r[:, None, None, None] ** 2))

    # bilateral filter
    bilateral_weight = gaussian_dist[None] * gaussian_value
    result_unfolded = torch.sum(bilateral_weight * image_unfolded, dim=-1)
    z_constant = bilateral_weight.sum(dim=-1)
    result_unfolded = result_unfolded / z_constant
    result = result_unfolded.transpose(1, 2).reshape(bsz, ch, height, width)
    return result


def ensure_tensor(x):
    if isinstance(x, torch.Tensor):
        return x
    # numpy
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x)
    return torch.tensor(x)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
