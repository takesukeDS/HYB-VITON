#!/usr/bin/env python
# coding=utf-8
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and

from multiprocessing import Pool
import argparse
import csv
import logging
import os
import os.path as osp
import time
import numpy as np
import torch
import torch.utils.checkpoint
import transformers
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm

import diffusers
from diffusers.utils import check_min_version, deprecate, is_wandb_available

from dataset import HybvtonFilterDataset, SdvtonFilterDataset, HrvitonFilterDataset
from config import cfg, update_config
from utils.image import bilateral_filter
import torch.nn.functional as F


# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.18.0")

logger = logging.getLogger(__name__)
logger.setLevel("INFO")

def make_image_grid(imgs, rows, cols):
    assert len(imgs) == rows * cols

    w, h = imgs[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid


def normalize_image(image):
    return (image - 0.5) * 2.0


def denormalize_image(image):
    return (image + 1.0) / 2.0


DATASET_TO_SUFFIXES = {
    "vitonhd": ("_00", "_00"),
    "dresscode": ("_0", "_1"),
}

def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)
    parser.add_argument(
        "process_type",
        type=str,
        choices=['split', 'filter'],
    )
    parser.add_argument(
        "--erode_kernel_size_w",
        type=int,
        default=21,
    )
    parser.add_argument(
        "--only_mask",
        action="store_true",
    )
    parser.add_argument(
        "--num_processes",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--unpaired",
        action="store_true",
    )
    parser.add_argument(
        "--pairs_path",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--pairs_path_prefix",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--phase",
        type=str,
        default="val",
    )
    parser.add_argument(
        "--warped_cloth",
        action="store_true",
    )
    parser.add_argument(
        "--warp_method",
        type=str,
        choices=["gpvton", "sdviton", "hrviton", "none"],
        default="none",
    )
    parser.add_argument(
        "--extract_torso",
        action="store_true",
    )
    parser.add_argument(
        "--warped_mask_format",
        type=str,
        default="pt",
    )
    parser.add_argument(
        "--save_format_mask",
        type=str,
        default="pt",
    )
    ############################################
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--mask_base_dir",
        type=str,
        default="hybvton_mask",
    )
    parser.add_argument(
        "--image_base_dir",
        type=str,
        default="ldvton_datasets",
    )
    parser.add_argument(
        "--pose_base_dir",
        type=str,
        default="densepose_segm",
    )
    parser.add_argument(
        "--parse_base_dir",
        type=str,
        default="schp_outputs",
    )
    parser.add_argument(
        "--hand_base_dir",
        type=str,
        default="hybvton_openpose_hand",
    )
    parser.add_argument(
        "--warped_cloth_base_dir",
        type=str,
        default="gpvton_warped_cloth_torso",
    )
    parser.add_argument(
        "--check_nan",
        action="store_true",
        help="fed to anomaly detection, if true runtime error is raised if nan is detected",
    )
    parser.add_argument(
        "--use_mini_val",
        action="store_true",
        help="use mini val set for validation",
    )
    parser.add_argument(
        "--dataset_cache_size",
        type=int,
        default=0,
        help="Number of samples to cache in memory for each dataset.",
    )
    parser.add_argument(
        "--reset_optimizer",
        action="store_true",
        help="reset optimizer",
    )
    parser.add_argument(
        "--best_val",
        type=float,
        default=None,
        help="best validation score, when provided validation before training is skipped",
    )
    # hybvton specific arguments above
    parser.add_argument(
        "--input_perturbation", type=float, default=0, help="The scale of input perturbation. Recommended 0.1."
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        ),
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="hybvton-preprocess",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--batch_size", type=int, default=4, help="Batch size (per device) for the evaluating dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--snr_gamma",
        type=float,
        default=None,
        help="SNR weighting gamma to be used if rebalancing the loss. Recommended value is 5.0. "
        "More details here: https://arxiv.org/abs/2303.09556.",
    )
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument("--use_ema", action="store_true", help="Whether to use EMA model.")
    parser.add_argument(
        "--non_ema_revision",
        type=str,
        default=None,
        required=False,
        help=(
            "Revision of pretrained non-ema model identifier. Must be a branch, tag or git identifier of the local or"
            " remote repository specified with --pretrained_model_name_or_path."
        ),
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--prediction_type",
        type=str,
        default=None,
        help="The prediction_type that shall be used for training. Choose between 'epsilon' or 'v_prediction' or leave `None`. If left to `None` the default prediction type of the scheduler: `noise_scheduler.config.prediciton_type` is chosen.",
    )
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=8000,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=1,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument("--noise_offset", type=float, default=0, help="The scale of noise offset.")
    parser.add_argument(
        "--validation_epochs",
        type=int,
        default=2,
        help="Run validation every X epochs.",
    )
    parser.add_argument(
        "--tracker_project_name",
        type=str,
        default="hybvton",
        help=(
            "The `project_name` argument passed to Accelerator.init_trackers for"
            " more information see https://huggingface.co/docs/accelerate/v0.17.0/en/package_reference/accelerator#accelerate.Accelerator"
        ),
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
    )

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    # default to using the same revision for the non-ema model if not specified
    if args.non_ema_revision is None:
        args.non_ema_revision = args.revision

    return args

def add_logging_hander(args):
    time_str = time.strftime('%Y-%m-%d-%H-%M')
    log_file = '{}_{}_{}.log'.format(
        os.path.splitext(os.path.basename(args.cfg))[0], time_str, "train")
    final_log_file = os.path.join(args.output_dir, log_file)
    handler = logging.FileHandler(final_log_file)
    file_formatter = logging.Formatter(
        fmt="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S"
    )
    handler.setFormatter(file_formatter)
    handler.setLevel(logging.INFO)
    logger.addHandler(handler)


def do_filter(process_index):
    global cfg
    args = parse_args()
    add_logging_hander(args)
    # loading configurations from yaml file
    update_config(cfg, args.cfg)

    if args.pairs_path_prefix is None:
        raise ValueError("pairs_path_prefix is required for filter process.")
    if args.non_ema_revision is not None:
        deprecate(
            "non_ema_revision!=None",
            "0.15.0",
            message=(
                "Downloading 'non_ema' weights from revision branches of the Hub is deprecated. Please make sure to"
                " use `--variant=non_ema` instead."
            ),
        )
    logging_dir = os.path.join(args.output_dir, args.logging_dir)

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    if process_index == 0:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    cloth_source_dir_name = "warped" if args.warped_cloth else "gt"
    paired_or_unpaired = "unpaired" if args.unpaired else "paired"
    image_save_dir = osp.join(
        args.output_dir, paired_or_unpaired, args.dataset_name, cloth_source_dir_name, "image", args.phase)
    mask_save_dir = osp.join(
        args.output_dir, paired_or_unpaired, args.dataset_name, cloth_source_dir_name, "mask", args.phase)
    os.makedirs(image_save_dir, exist_ok=True)
    os.makedirs(mask_save_dir, exist_ok=True)
    ext = ".txt"
    prefix = args.pairs_path_prefix
    pairs_path = osp.join(args.output_dir, prefix + str(process_index) + ext)

    # Preprocessing the datasets.
    person_transforms = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Lambda(normalize_image),
        ]
    )
    cloth_transforms = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Lambda(normalize_image),
        ]
    )
    dataset_name = args.dataset_name
    suffix_p, suffix_c = DATASET_TO_SUFFIXES[dataset_name]
    assert len(cfg.DATASET.ERODE_KERNEL_SIZES) == 2
    erode_kernel_size = cfg.DATASET.ERODE_KERNEL_SIZES[0] if dataset_name == "vitonhd" else cfg.DATASET.ERODE_KERNEL_SIZES[1]
    dataset_class = HybvtonFilterDataset
    if args.warp_method == "gpvton":
        dataset_class = HybvtonFilterDataset
    elif args.warp_method == "sdviton":
        dataset_class = SdvtonFilterDataset
    elif args.warp_method == "hrviton":
        dataset_class = HrvitonFilterDataset
    cur_dataset = dataset_class(
        args.image_base_dir, args.mask_base_dir, args.parse_base_dir, args.hand_base_dir,
        args.pose_base_dir, args.warped_cloth_base_dir,
        dataset_name, args.phase, pairs_path,
        person_transforms=person_transforms, cloth_transforms=cloth_transforms,
        max_rotation_degree=0, max_rotation_degree_cloth=0, random_flip=False,
        person_image_suffix=suffix_p, cloth_image_suffix=suffix_c,
        resize_shape=cfg.DATASET.RESIZE_SHAPE, cache_size=0, use_gt_warped_cloth=not args.warped_cloth,
        extract_torso=args.extract_torso, warped_mask_format=args.warped_mask_format,
    )
    if args.use_mini_val:
        cur_dataset = torch.utils.data.Subset(cur_dataset, range(0, len(cur_dataset), 10))
    cur_dataloader = torch.utils.data.DataLoader(
        cur_dataset,
        batch_size=args.batch_size,
        num_workers=args.dataloader_num_workers,
        shuffle=False,
    )

    device = torch.device("cuda" if torch.cuda.is_available() and args.gpu else "cpu")

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(len(cur_dataloader)), disable=process_index != 0)
    progress_bar.set_description("Steps")
    erode_kernel_size = (erode_kernel_size,
                         args.erode_kernel_size_w) if args.erode_kernel_size_w is not None else erode_kernel_size
    erode_padding = erode_kernel_size // 2 if isinstance(erode_kernel_size, int) else (
    erode_kernel_size[0] // 2, erode_kernel_size[1] // 2)
    for batch in cur_dataloader:
        warped_mask = batch["mask_warped"].to(device)
        warped_cloth = batch["image_torso"].to(device)

        if not args.only_mask:
            warped_cloth = batch["image_torso"].to(device)
            for _ in range(cfg.PREPROCESS.BILATERAL_FILTER_ITERATIONS):
                warped_cloth = bilateral_filter(warped_cloth, cfg.PREPROCESS.BILATERAL_KERNEL_SIZE,
                                                        cfg.PREPROCESS.BILATERAL_SIGMA_D,
                                                        cfg.PREPROCESS.BILATERAL_SIGMA_R)
        # since clothing image is not guaranteed to contain all pixels to construct warped clothing,
        # explicit transformation is not always sound due to shape constraints
        # therefore we decided to use only internal region of warped garment.
        if cfg.DATASET.NUM_ERODE_ITERATIONS > 0:
            for _ in range(cfg.DATASET.NUM_ERODE_ITERATIONS):
                warped_mask = 1 - F.max_pool2d(1 - warped_mask, erode_kernel_size, stride=1,
                                                    padding=erode_padding)
                warped_cloth = warped_cloth * warped_mask + \
                               torch.zeros_like(warped_cloth) * (1 - warped_mask)
        cloth_ids = [x.split("_")[0] + suffix_c for x in batch["id_cloth"]]
        person_ids = [x.split("_")[0] + suffix_p for x in batch["id"]]
        for i, person_id, cloth_id in zip(range(len(batch["id"])), person_ids, cloth_ids):
            if not args.only_mask:
                torso_image_filtered_np = (denormalize_image(
                    warped_cloth[i]).cpu().numpy().transpose(1, 2, 0) * 255).round().astype(np.uint8)
                torso_image_filtered_pil = Image.fromarray(torso_image_filtered_np)
                torso_image_filtered_pil.save(
                    os.path.join(image_save_dir, person_id + "_" + cloth_id + ".png"))
            if args.save_format_mask == "pt":
                torch.save(warped_mask[i], os.path.join(mask_save_dir, person_id + "_" + cloth_id + ".pt"))
            elif args.save_format_mask == "png":
                mask_warped_np = (warped_mask[i].cpu().numpy() > 0.5)
                mask_warped_pil = Image.fromarray(mask_warped_np[0])
                mask_warped_pil.save(os.path.join(mask_save_dir, person_id + "_" + cloth_id + ".png"))
            else:
                raise ValueError("save_format_mask must be pt or png.")
        if process_index == 0:
            progress_bar.update(1)


def split_pairs(args):
    if args.seed is not None:
        torch.manual_seed(args.seed)
    if args.pairs_path is None:
        raise ValueError("pairs_path must be provided.")

    prefix = osp.splitext(osp.basename(args.pairs_path))[0] + "_"
    ext = ".txt"
    with open(args.pairs_path, "r") as f:
        reader = csv.reader(f, delimiter=" ")
        rows = list(reader)

    for process_id in range(args.num_processes):
        with open(osp.join(args.output_dir, prefix + str(process_id) + ext), "w") as f:
            writer = csv.writer(f, delimiter=" ")
            for i, row in enumerate(rows):
                if i % args.num_processes == process_id:
                    writer.writerow(row)

if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    # Load the model and tokenizer.
    logger.info("***** Running preprocess *****")
    logger.info(f"  Process type = {args.process_type}")
    logger.info(f"  Dataset name = {args.dataset_name}")
    logger.info(f"  Current phase = {args.phase}")
    logger.info(f"  Batch size = {args.batch_size}")
    logger.info(f"  Num processes = {args.num_processes}")
    if args.process_type == "split":
        split_pairs(args)
    elif args.process_type == "filter":
        if args.num_processes == 1:
            do_filter(0)
        else:
            if args.pairs_path_prefix is None:
                raise ValueError("pairs_path_prefix must be provided.")
            paired_or_unpaired = "unpaired" if args.unpaired else "paired"
            cloth_source_dir_name = "warped" if args.warped_cloth else "gt"
            image_save_dir = osp.join(args.output_dir, paired_or_unpaired, args.dataset_name, cloth_source_dir_name,
                                      args.phase)
            os.makedirs(image_save_dir, exist_ok=True)
            num_processes = args.num_processes
            with Pool(num_processes) as p:
                print(p.map(do_filter, range(num_processes)))
    print("Done")