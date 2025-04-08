from __future__ import division, absolute_import, print_function

import torch
from torch import nn


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
