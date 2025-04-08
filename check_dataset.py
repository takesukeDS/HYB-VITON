import os
from os.path import join as opj
from omegaconf import OmegaConf
from importlib import import_module
import argparse

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.utils import make_grid

from cldm.plms_hacked import PLMSSampler
from cldm.model import create_model
from utils import tensor2img
from PIL import Image

def build_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--data_root_dir", type=str, default="./DATA/zalando-hd-resized")
    parser.add_argument("--unpair", action="store_true")
    parser.add_argument("--save_dir", type=str, default="./dataset_check")

    parser.add_argument("--img_H", type=int, default=512)
    parser.add_argument("--img_W", type=int, default=384)
    parser.add_argument("--eta", type=float, default=0.0)

    # Hybvton
    parser.add_argument("--phase", type=str)

    args = parser.parse_args()
    return args


@torch.no_grad()
def main(args):
    batch_size = args.batch_size
    img_H = args.img_H
    img_W = args.img_W

    config = OmegaConf.load(args.config_path)
    config.model.params.img_H = args.img_H
    config.model.params.img_W = args.img_W
    params = config.model.params

    dataset = getattr(import_module("dataset"), config.dataset_name)(
        data_root_dir=args.data_root_dir,
        img_H=img_H,
        img_W=img_W,
        phase=args.phase,
        is_paired=not args.unpair,
        is_sorted=True
    )
    dataloader = DataLoader(dataset, num_workers=4, shuffle=False, batch_size=batch_size, pin_memory=True)

    shape = (4, img_H//8, img_W//8)
    save_dir = opj(args.save_dir, "unpair" if args.unpair else "pair")
    os.makedirs(save_dir, exist_ok=True)
    def denormalize(x):
        return (x + 1) / 2

    def to_rgb(x):
        return (x * 255).to(torch.uint8)

    for batch_idx, batch in enumerate(dataloader):
        for i in range(batch_size):
            grid = torch.cat([
                to_rgb(denormalize(batch["agn"][i])),
                to_rgb(batch["agn_mask"][i].expand(-1, -1, 3)),
                to_rgb(denormalize(batch["cloth"][i])),
                to_rgb(batch["cloth_mask"][i].expand(-1, -1, 3)),
                to_rgb(denormalize(batch["image"][i])),
                to_rgb(denormalize(batch["image_densepose"][i])),
                to_rgb(denormalize(batch["hybvton_warped_cloth"][i])),
                to_rgb(batch["hybvton_warped_mask"][i].expand(-1, -1, 3)),
                to_rgb(batch["gt_cloth_warped_mask"][i].expand(-1, -1, 3)),
            ], dim=1)
            Image.fromarray(grid.numpy()).save(opj(save_dir, f"{batch_idx * batch_size + i}.png"))
        break

if __name__ == "__main__":
    args = build_args()
    main(args)
