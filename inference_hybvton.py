import os
from functools import partial
from os.path import join as opj
from omegaconf import OmegaConf
from importlib import import_module
import argparse

import cv2
import numpy as np
import torch
from torch.nn import GroupNorm
from torch.utils.data import DataLoader

from cldm.plms_hacked import PLMSSampler, PLMSSamplerHybvton
from cldm.model import create_model
from networks import ConditionGenerator, load_checkpoint
from utils import tensor2img, set_seed, bilateral_filter
import torch.nn.functional as F


def build_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str)
    parser.add_argument("--model_load_path", type=str)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--data_root_dir", type=str, default="./DATA/zalando-hd-resized")
    parser.add_argument("--repaint", action="store_true")
    parser.add_argument("--unpair", action="store_true")
    parser.add_argument("--save_dir", type=str, default="./samples")

    parser.add_argument("--denoise_steps", type=int, default=50)
    parser.add_argument("--img_H", type=int, default=512)
    parser.add_argument("--img_W", type=int, default=384)
    parser.add_argument("--eta", type=float, default=0.0)

    # Hybvton
    parser.add_argument("--phase", type=str)
    parser.add_argument("--latent_nc", type=int, default=4)
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="runwayml/stable-diffusion-v1-5",
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument("--bilateral_kernel_size", type=int, default=23)
    parser.add_argument("--bilateral_sigma_d", type=float, default=5.0)
    parser.add_argument("--bilateral_sigma_r", type=float, default=0.06)
    parser.add_argument("--bilateral_filter_iterations", type=int, default=4)
    parser.add_argument("--num_erode_iterations", type=int, default=1)
    parser.add_argument("--erode_kernel_size", type=int, default=21)
    parser.add_argument("--erode_kernel_size_w", type=int, default=None)
    parser.add_argument("--timestep_threshold", type=int, default=1000)
    parser.add_argument('--tocg_checkpoint', type=str, default=None, help='tocg checkpoint')
    parser.add_argument("--display_cond", action="store_true")
    parser.add_argument("--extract_torso", action="store_true")
    parser.add_argument("--torso_extraction_method", choices=['arm_elimination', 'torso_segment', 'none'],
                        default='none')
    parser.add_argument("--save_dir_cond", type=str, default="display_cond")
    parser.add_argument("--use_preprocessed", action="store_true")
    parser.add_argument("--only_one_refinement", action="store_true")
    parser.add_argument("--start_from_noised_agn", action="store_true")
    parser.add_argument("--seed", type=int, default=1235)
    parser.add_argument("--use_hybvton_densepose_torso", action="store_true")

    # GAN network
    parser.add_argument("--warp_feature", choices=['encoder', 'T1'], default="T1")
    parser.add_argument("--out_layer", choices=['relu', 'conv'], default="relu")
    parser.add_argument("--semantic_nc", type=int, default=13)
    parser.add_argument("--output_nc", type=int, default=13)

    args = parser.parse_args()
    return args

def build_opt(args):
    opt = argparse.Namespace()
    opt.dataroot = args.data_root_dir
    opt.warp_feature = args.warp_feature
    opt.out_layer = args.out_layer
    opt.semantic_nc = args.semantic_nc
    opt.output_nc = args.output_nc
    opt.latent_nc = args.latent_nc
    opt.cuda = True

    return opt


def build_config(args, config_path=None):
    if config_path is None:
        config_path = args.config_path
    config = OmegaConf.load(config_path)
    config.model.params.img_H = args.img_H
    config.model.params.img_W = args.img_W
    return config


@torch.no_grad()
def main(args):
    opt = build_opt(args)
    batch_size = args.batch_size
    img_H = args.img_H
    img_W = args.img_W

    config = build_config(args)
    params = config.model.params

    model = create_model(config_path=None, config=config)
    load_cp = torch.load(args.model_load_path, map_location="cpu")
    load_cp = load_cp["state_dict"] if "state_dict" in load_cp.keys() else load_cp
    model.load_state_dict(load_cp)
    model = model.cuda()
    model.eval()

    tocg = None
    if args.tocg_checkpoint:
        input1_nc = 4  # cloth + cloth-mask
        input2_nc = opt.semantic_nc + 3 + args.latent_nc  # parse_agnostic + densepose + latents(diffusion model)
        norm_class = partial(GroupNorm, 32)
        tocg = ConditionGenerator(
            opt, input1_nc=input1_nc, input2_nc=input2_nc, output_nc=opt.output_nc, ngf=96, norm_layer=norm_class)
        # Load Checkpoint
        load_checkpoint(tocg, args.tocg_checkpoint)
        tocg = tocg.cuda()
        tocg.eval()

    if tocg is not None and args.torso_extraction_method != 'none':
        print("WARNING: torso extraction method is not compatible with tocg (i.e. refinement)")

    sampler = PLMSSamplerHybvton(model, bilateral_kernel_size=args.bilateral_kernel_size,
                                  bilateral_sigma_d=args.bilateral_sigma_d, bilateral_sigma_r=args.bilateral_sigma_r,
                                  bilateral_filter_iterations=args.bilateral_filter_iterations,
                                  num_erode_iterations=args.num_erode_iterations,
                                 erode_kernel_size=args.erode_kernel_size, timestep_threshold=args.timestep_threshold,
                                 tocg=tocg, display_cond=args.display_cond, extract_torso=args.extract_torso,
                                 save_dir_cond=args.save_dir_cond, use_preprocessed=args.use_preprocessed,
                                 only_one_refinement=args.only_one_refinement)
    dataset = getattr(import_module("dataset"), config.dataset_name)(
        data_root_dir=args.data_root_dir,
        img_H=img_H,
        img_W=img_W,
        phase=args.phase,
        is_paired=not args.unpair,
        is_sorted=True,
        semantic_nc=opt.semantic_nc,
        use_preprocessed=args.use_preprocessed,
        torso_extraction_method=args.torso_extraction_method,
    )
    dataloader = DataLoader(dataset, num_workers=4, shuffle=False, batch_size=batch_size, pin_memory=True)

    shape = (4, img_H//8, img_W//8) 
    save_dir = opj(args.save_dir, "unpair" if args.unpair else "pair")
    os.makedirs(save_dir, exist_ok=True)
    erode_kernel_size = (args.erode_kernel_size, args.erode_kernel_size_w) if args.erode_kernel_size_w is not None else args.erode_kernel_size
    erode_padding = erode_kernel_size // 2 if isinstance(erode_kernel_size, int) else (erode_kernel_size[0] // 2, erode_kernel_size[1] // 2)
    for batch_idx, batch in enumerate(dataloader):
        print(f"{batch_idx}/{len(dataloader)}")
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.cuda()
        if args.torso_extraction_method != 'none':
            warped_cloth = batch["hybvton_warped_cloth"].permute(0, 3, 1, 2)
            warped_clothmask = batch["hybvton_warped_mask"].permute(0, 3, 1, 2)
            if args.num_erode_iterations > 0:
                for _ in range(args.num_erode_iterations):
                    warped_clothmask = 1 - F.max_pool2d(1 - warped_clothmask, erode_kernel_size, stride=1,
                                                        padding=erode_padding)
                warped_cloth = warped_cloth * warped_clothmask + \
                               torch.zeros_like(warped_cloth) * (1 - warped_clothmask)

            if args.bilateral_filter_iterations > 0:
                for _ in range(args.bilateral_filter_iterations):
                    warped_cloth = bilateral_filter(warped_cloth, args.bilateral_kernel_size,
                                                    args.bilateral_sigma_d, args.bilateral_sigma_r)
            warped_cloth = warped_cloth.permute(0, 2, 3, 1)
            warped_clothmask = warped_clothmask.permute(0, 2, 3, 1)
            batch["agn"] = batch["agn_orig"] * (1 - warped_clothmask) + warped_cloth * warped_clothmask
            batch["agn_mask"] = (batch["agn_mask_orig"] + warped_clothmask).clip(0, 1)
            batch["hybvton_warped_mask"] = warped_clothmask

        z, c = model.get_input(batch, params.first_stage_key)
        bs = z.shape[0]
        c_crossattn = c["c_crossattn"][0][:bs]
        if c_crossattn.ndim == 4:
            c_crossattn = model.get_learned_conditioning(c_crossattn)
            c["c_crossattn"] = [c_crossattn]
        uc_cross = model.get_unconditional_conditioning(bs)
        uc_full = {"c_concat": c["c_concat"], "c_crossattn": [uc_cross]}
        uc_full["first_stage_cond"] = c["first_stage_cond"]
        sampler.model.batch = batch
        ts = torch.full((1,), 999, device=z.device, dtype=torch.long)

        start_code = None
        if args.start_from_noised_agn:
            start_code = model.q_sample(c["first_stage_cond"][:, :4], ts)

        samples, _, _ = sampler.sample(
            batch,
            args.denoise_steps,
            bs,
            shape, 
            c,
            x_T=start_code,
            verbose=False,
            eta=args.eta,
            unconditional_conditioning=uc_full,
        )

        x_samples = model.decode_first_stage(samples)
        for sample_idx, (x_sample, fn,  cloth_fn) in enumerate(zip(x_samples, batch['img_fn'], batch["cloth_fn"])):
            x_sample_img = tensor2img(x_sample, round=True)  # [0, 255]
            if args.repaint:
                repaint_agn_img = np.uint8((batch["image"][sample_idx].cpu().numpy()+1)/2 * 255 + 0.5)   # [0,255]
                repaint_agn_mask_img = batch["agn_mask_orig"][sample_idx].cpu().numpy()  # 0 or 1
                x_sample_img = repaint_agn_img * repaint_agn_mask_img + x_sample_img * (1-repaint_agn_mask_img)
                x_sample_img = np.uint8(x_sample_img + 0.5)

            to_path = opj(save_dir, f"{fn.split('.')[0]}_{cloth_fn.split('.')[0]}.jpg")
            cv2.imwrite(to_path, x_sample_img[:,:,::-1])

if __name__ == "__main__":
    args = build_args()
    set_seed(args.seed)
    main(args)
