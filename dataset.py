import os
from os.path import join as opj

import cv2
import numpy as np
import albumentations as A
import torch
from torch.utils.data import Dataset
from PIL import Image
import os.path as osp

from torchvision.transforms import InterpolationMode
import torchvision.transforms.functional as TF
from torchvision import transforms


def imread(
        p, h, w, 
        is_mask=False, 
        in_inverse_mask=False, 
        img=None
):
    if img is None:
        img = cv2.imread(p)
    if not is_mask:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (w,h))
        img = (img.astype(np.float32) / 127.5) - 1.0  # [-1, 1]
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (w,h))
        img = (img >= 128).astype(np.float32)  # 0 or 1
        img = img[:,:,None]
        if in_inverse_mask:
            img = 1-img
    return img

def imread_for_albu(
        p, 
        is_mask=False, 
        in_inverse_mask=False, 
        cloth_mask_check=False, 
        use_resize=False, 
        height=512, 
        width=384,
):
    img = cv2.imread(p)
    if use_resize:
        img = cv2.resize(img, (width, height))
    if not is_mask:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = (img>=128).astype(np.float32)
        if cloth_mask_check:
            if img.sum() < 30720*4:
                img = np.ones_like(img).astype(np.float32)
        if in_inverse_mask:
            img = 1 - img
        img = np.uint8(img*255.0)
    return img
def norm_for_albu(img, is_mask=False):
    if not is_mask:
        img = (img.astype(np.float32)/127.5) - 1.0
    else:
        img = img.astype(np.float32) / 255.0
        img = img[:,:,None]
    return img


DENSEPOSE_SEGM_RGB_TORSO = [ 20,  80, 194]
DENSEPOSE_SEGM_RGB_RIGHT_ARM = [
    [170, 189, 105], #right_arm_upper_inside
    [216, 186,  86], #right_arm_upper_outside
    [240, 199,  60], #right_arm_lower_inside
    [251, 220,  36], #right_arm_lower_outside
]
DENSEPOSE_SEGM_RGB_LEFT_ARM = [
    [145, 191, 116], #left_arm_upper_inside
    [192, 188,  96], #left_arm_upper_outside
    [228, 192,  74], #left_arm_lower_inside
    [252, 206,  46], #left_arm_lower_outside
]


DENSEPOSE_SEGM_RGB_RIGHT_ARM_RED = [
    170, 216, 240, 251
]
DENSEPOSE_SEGM_RGB_LEFT_ARM_RED = [
    145, 192, 228, 252
]
DENSEPOSE_SEGM_RGB_TORSO_RED = [
    20
]


def densepose_to_armmask(segm_np):
    segm_np_red = segm_np[:, :, 0]
    mask_arm = np.isin(segm_np_red, DENSEPOSE_SEGM_RGB_RIGHT_ARM_RED + DENSEPOSE_SEGM_RGB_LEFT_ARM_RED)
    return mask_arm


class VITONHDDataset(Dataset):
    def __init__(
            self, 
            data_root_dir, 
            img_H, 
            img_W,
            phase,
            is_paired=True, 
            is_sorted=False,
            transform_size=None, 
            transform_color=None,
            torso_extraction_method="none",
            **kwargs
        ):
        self.drd = data_root_dir
        self.img_H = img_H
        self.img_W = img_W
        self.pair_key = "paired" if is_paired else "unpaired"
        self.data_type = "train" if phase in ["train", "val"] else "test"
        self.is_test = phase in ["val", "test"]
        self.phase = phase
        self.resize_ratio_H = 1.0
        self.resize_ratio_W = 1.0
        self.torso_extraction_method = torso_extraction_method
        self.use_hybvton_densepose_torso = kwargs.get("use_hybvton_densepose_torso", False)

        self.resize_transform = A.Resize(img_H, img_W)
        self.transform_size = None
        self.transform_crop_person = None
        self.transform_crop_cloth = None
        self.transform_color = None

        #### spatial aug >>>>
        transform_crop_person_lst = []
        transform_crop_cloth_lst = []
        transform_size_lst = [A.Resize(int(img_H*self.resize_ratio_H), int(img_W*self.resize_ratio_W))]
        transform_hflip_lst = []
    
        if transform_size is not None:
            if "hflip" in transform_size:
                transform_hflip_lst.append(A.HorizontalFlip(p=0.5))

            if "shiftscale" in transform_size:
                transform_crop_person_lst.append(A.ShiftScaleRotate(rotate_limit=0, shift_limit=0.2, scale_limit=(-0.2, 0.2), border_mode=cv2.BORDER_CONSTANT, p=0.5, value=0))
                transform_crop_cloth_lst.append(A.ShiftScaleRotate(rotate_limit=0, shift_limit=0.2, scale_limit=(-0.2, 0.2), border_mode=cv2.BORDER_CONSTANT, p=0.5, value=0))

        self.transform_crop_person = A.Compose(
                transform_crop_person_lst,
                additional_targets={"agn":"image", 
                                    "agn_mask":"image", 
                                    "cloth_mask_warped":"image", 
                                    "cloth_warped":"image", 
                                    "image_densepose":"image", 
                                    "image_parse":"image", 
                                    "gt_cloth_warped_mask":"image",
                                    "hybvton_warped_cloth": "image",
                                    "hybvton_warped_mask": "image",
                                    }
        )
        self.transform_crop_cloth = A.Compose(
                transform_crop_cloth_lst,
                additional_targets={"cloth_mask":"image"}
        )

        self.transform_size = A.Compose(
                transform_size_lst,
                additional_targets={"agn":"image", 
                                    "agn_mask":"image", 
                                    "cloth":"image", 
                                    "cloth_mask":"image", 
                                    "cloth_mask_warped":"image", 
                                    "cloth_warped":"image", 
                                    "image_densepose":"image", 
                                    "image_parse":"image", 
                                    "gt_cloth_warped_mask":"image",
                                    "image_densepose_hybvton":"image",
                                    }
            )
        self.transform_hflip = A.Compose(
                transform_hflip_lst,
                additional_targets={"agn":"image",
                                    "agn_mask":"image",
                                    "cloth":"image",
                                    "cloth_mask":"image",
                                    "cloth_mask_warped":"image",
                                    "cloth_warped":"image",
                                    "image_densepose":"image",
                                    "image_parse":"image",
                                    "gt_cloth_warped_mask":"image",
                                    "hybvton_warped_cloth": "image",
                                    "hybvton_warped_mask": "image",
                                    }
            )
        #### spatial aug <<<<

        #### non-spatial aug >>>>
        if transform_color is not None:
            transform_color_lst = []
            for t in transform_color:
                if t == "hsv":
                    transform_color_lst.append(A.HueSaturationValue(5,5,5,p=0.5))
                elif t == "bright_contrast":
                    transform_color_lst.append(A.RandomBrightnessContrast(brightness_limit=(-0.1, 0.02), contrast_limit=(-0.3, 0.3), p=0.5))

            self.transform_color = A.Compose(
                transform_color_lst,
                additional_targets={"agn":"image", 
                                    "cloth":"image",  
                                    "cloth_warped":"image",
                                    }
            )
        #### non-spatial aug <<<<
                    
        assert not (self.phase == "train" and self.pair_key == "unpaired"), f"train must use paired dataset"
        
        im_names = []
        c_names = []
        with open(opj(self.drd, f"hybvton_{self.phase}_pairs.txt"), "r") as f:
            for line in f.readlines():
                im_name, c_name = line.strip().split()
                im_names.append(im_name)
                c_names.append(c_name)
        if is_sorted:
            im_names, c_names = zip(*sorted(zip(im_names, c_names)))
        self.im_names = im_names
        
        self.c_names = dict()
        self.c_names["paired"] = im_names
        self.c_names["unpaired"] = c_names

    def __len__(self):
        return len(self.im_names)
    
    def __getitem__(self, idx):
        img_fn = self.im_names[idx]
        cloth_fn = self.c_names[self.pair_key][idx]
        if self.transform_size is None and self.transform_color is None:
            raise NotImplementedError("Never reached by original code")
            agn = imread(
                opj(self.drd, self.data_type, "agnostic-v3.2", self.im_names[idx]), 
                self.img_H, 
                self.img_W
            )
            agn_mask = imread(
                opj(self.drd, self.data_type, "agnostic-mask", self.im_names[idx].replace(".jpg", "_mask.png")), 
                self.img_H, 
                self.img_W, 
                is_mask=True, 
            )
            cloth = imread(
                opj(self.drd, self.data_type, "cloth", self.c_names[self.pair_key][idx]), 
                self.img_H, 
                self.img_W
            )

            gt_cloth_warped_mask = imread(
                opj(self.drd, self.data_type, "gt_cloth_warped_mask", self.im_names[idx]), 
                self.img_H, 
                self.img_W, 
                is_mask=True
            ) if not self.is_test else np.zeros_like(agn_mask)

            hybvton_warped_cloth = imread(
                opj(self.drd, self.data_type, "hybvton_warped_cloth_" + self.pair_key,
                    self.im_names[idx].split(".")[0] + "_" + self.c_names[self.pair_key][idx].replace(".jpg", ".png")),
                self.img_H,
                self.img_W,
            )
            hybvton_warped_mask = imread(
                opj(self.drd, self.data_type, "hybvton_warped_mask_" + self.pair_key,
                    self.im_names[idx].split(".")[0] + "_" + self.c_names[self.pair_key][idx].replace(".jpg", ".png")),
                self.img_H,
                self.img_W,
                is_mask=True
            )

            image = imread(opj(self.drd, self.data_type, "image", self.im_names[idx]), self.img_H, self.img_W)
            image_densepose = imread(opj(self.drd, self.data_type, "image-densepose", self.im_names[idx]), self.img_H, self.img_W)
            hybvton_warped_mask = (hybvton_warped_mask / 255 * agn_mask / 255)
            agn = agn * (1 - hybvton_warped_mask[:,:,None]) + hybvton_warped_cloth * hybvton_warped_mask[:,:,None]
            agn = agn.astype(np.uint8)
            hybvton_warped_mask = (hybvton_warped_mask * 255).astype(np.uint8)
            agn_mask_orig = 255 - agn_mask
            agn_mask = np.clip(agn_mask - hybvton_warped_mask, 0, 255)
            agn_mask = 255 - agn_mask

        else:
            agn = imread_for_albu(opj(self.drd, self.data_type, "agnostic-v3.2", self.im_names[idx]))
            agn_mask = imread_for_albu(opj(self.drd, self.data_type, "agnostic-mask", self.im_names[idx].replace(".jpg", "_mask.png")), is_mask=True)
            cloth = imread_for_albu(opj(self.drd, self.data_type, "cloth", self.c_names[self.pair_key][idx]))
            cloth_mask = imread_for_albu(opj(self.drd, self.data_type, "cloth-mask", self.c_names[self.pair_key][idx]), is_mask=True, cloth_mask_check=True)
            
            gt_cloth_warped_mask = imread_for_albu(
                opj(self.drd, self.data_type, "gt_cloth_warped_mask", self.im_names[idx]),
                is_mask=True
            ) if not self.is_test else np.zeros_like(agn_mask)
            hybvton_warped_cloth = imread_for_albu(
                opj(self.drd, self.data_type, "hybvton_warped_cloth_" + self.pair_key,
                    self.im_names[idx].split(".")[0] + "_" + self.c_names[self.pair_key][idx].replace(".jpg", ".png")),
            )
            hybvton_warped_mask = imread_for_albu(
                opj(self.drd, self.data_type, "hybvton_warped_mask_" + self.pair_key,
                    self.im_names[idx].split(".")[0] + "_" + self.c_names[self.pair_key][idx].replace(".jpg", ".png")),
                is_mask=True
            )
                
            image = imread_for_albu(opj(self.drd, self.data_type, "image", self.im_names[idx]))
            image_densepose = imread_for_albu(opj(self.drd, self.data_type, "image-densepose", self.im_names[idx]))
            if self.use_hybvton_densepose_torso:
                image_densepose_hybvton = imread_for_albu(
                    opj(self.drd, self.data_type, "image-densepose_hybvton", self.im_names[idx].replace(".jpg", ".png")))

            if self.transform_size is not None:
                kwargs_for_transform = dict(
                    image=image,
                    agn=agn,
                    agn_mask=agn_mask,
                    cloth=cloth,
                    cloth_mask=cloth_mask,
                    image_densepose=image_densepose,
                    gt_cloth_warped_mask=gt_cloth_warped_mask,
                )
                if self.use_hybvton_densepose_torso:
                    kwargs_for_transform["image_densepose_hybvton"] = image_densepose_hybvton
                transformed = self.transform_size(**kwargs_for_transform)
                image=transformed["image"]
                agn=transformed["agn"]
                agn_mask=transformed["agn_mask"]
                image_densepose=transformed["image_densepose"]
                gt_cloth_warped_mask=transformed["gt_cloth_warped_mask"]
                if self.use_hybvton_densepose_torso:
                    image_densepose_hybvton=transformed["image_densepose_hybvton"]

                cloth=transformed["cloth"]
                cloth_mask=transformed["cloth_mask"]

            hybvton_warped_mask = (hybvton_warped_mask / 255 * agn_mask / 255)
            agn_mask_orig = 255 - agn_mask
            agn_orig = agn

            if self.torso_extraction_method != "none":
                densepose_for_torso = image_densepose_hybvton if self.use_hybvton_densepose_torso else image_densepose
                if self.torso_extraction_method == "torso_segment":
                    densepose_torso_mask = densepose_for_torso[:, :, 0] == DENSEPOSE_SEGM_RGB_TORSO[0]
                    densepose_torso_mask = densepose_torso_mask.astype(np.float32)
                    hybvton_warped_mask = hybvton_warped_mask * densepose_torso_mask
                elif self.torso_extraction_method == "arm_elimination":
                    arm_mask = densepose_to_armmask(densepose_for_torso).astype(np.float32)
                    hybvton_warped_mask = hybvton_warped_mask * (1 - arm_mask)

            agn = agn * (1 - hybvton_warped_mask[:, :, None]) + hybvton_warped_cloth * hybvton_warped_mask[:, :, None]
            agn = agn.astype(np.uint8)
            hybvton_warped_mask = (hybvton_warped_mask * 255).astype(np.uint8)
            agn_mask = np.clip(agn_mask - hybvton_warped_mask, 0, 255)

            if self.transform_hflip is not None:
                transformed = self.transform_hflip(
                    image=image,
                    agn=agn,
                    agn_mask=agn_mask,
                    cloth=cloth,
                    cloth_mask=cloth_mask,
                    image_densepose=image_densepose,
                    gt_cloth_warped_mask=gt_cloth_warped_mask,
                    hybvton_warped_cloth=hybvton_warped_cloth,
                    hybvton_warped_mask=hybvton_warped_mask,
                )

                image=transformed["image"]
                agn=transformed["agn"]
                agn_mask=transformed["agn_mask"]
                image_densepose=transformed["image_densepose"]
                gt_cloth_warped_mask=transformed["gt_cloth_warped_mask"]
                hybvton_warped_cloth=transformed["hybvton_warped_cloth"]
                hybvton_warped_mask=transformed["hybvton_warped_mask"]

                cloth=transformed["cloth"]
                cloth_mask=transformed["cloth_mask"]

            if self.transform_crop_person is not None:
                transformed_image = self.transform_crop_person(
                    image=image,
                    agn=agn,
                    agn_mask=agn_mask,
                    image_densepose=image_densepose,
                    gt_cloth_warped_mask=gt_cloth_warped_mask,
                    hybvton_warped_cloth=hybvton_warped_cloth,
                    hybvton_warped_mask=hybvton_warped_mask,
                )

                image=transformed_image["image"]
                agn=transformed_image["agn"]
                agn_mask=transformed_image["agn_mask"]
                image_densepose=transformed_image["image_densepose"]
                gt_cloth_warped_mask=transformed_image["gt_cloth_warped_mask"]
                hybvton_warped_cloth=transformed_image["hybvton_warped_cloth"]
                hybvton_warped_mask=transformed_image["hybvton_warped_mask"]

            if self.transform_crop_cloth is not None:
                transformed_cloth = self.transform_crop_cloth(
                    image=cloth,
                    cloth_mask=cloth_mask
                )

                cloth=transformed_cloth["image"]
                cloth_mask=transformed_cloth["cloth_mask"]

            agn_mask = 255 - agn_mask
            if self.transform_color is not None:
                transformed = self.transform_color(
                    image=image, 
                    agn=agn, 
                    cloth=cloth,
                )

                image=transformed["image"]
                agn=transformed["agn"]
                cloth=transformed["cloth"]

                agn = agn * agn_mask[:,:,None].astype(np.float32)/255.0 + 128 * (1 - agn_mask[:,:,None].astype(np.float32)/255.0)
                
            agn = norm_for_albu(agn)
            agn_orig = norm_for_albu(agn_orig)
            agn_mask_orig = norm_for_albu(agn_mask_orig, is_mask=True)
            agn_mask_orig = (agn_mask_orig > 0.5).astype(np.float32)
            agn_mask = norm_for_albu(agn_mask, is_mask=True)
            cloth = norm_for_albu(cloth)
            cloth_mask = norm_for_albu(cloth_mask, is_mask=True)
            image = norm_for_albu(image)
            image_densepose = norm_for_albu(image_densepose)
            gt_cloth_warped_mask = norm_for_albu(gt_cloth_warped_mask, is_mask=True)
            hybvton_warped_cloth = norm_for_albu(hybvton_warped_cloth)
            hybvton_warped_mask = norm_for_albu(hybvton_warped_mask, is_mask=True)
            
        return dict(
            agn=agn,
            agn_orig=agn_orig,
            agn_mask=agn_mask,
            agn_mask_orig=agn_mask_orig,
            cloth=cloth,
            cloth_mask=cloth_mask,
            image=image,
            image_densepose=image_densepose,
            gt_cloth_warped_mask=gt_cloth_warped_mask,
            txt="",
            img_fn=img_fn,
            cloth_fn=cloth_fn,
            hybvton_warped_cloth=hybvton_warped_cloth,
            hybvton_warped_mask=hybvton_warped_mask,
        )


class VITONHDDatasetWithGAN(VITONHDDataset):
    # parse map
    labels = {
        0: ['background', [0, 10]],
        1: ['hair', [1, 2]],
        2: ['face', [4, 13]],
        3: ['upper', [5, 6, 7]],
        4: ['bottom', [9, 12]],
        5: ['left_arm', [14]],
        6: ['right_arm', [15]],
        7: ['left_leg', [16]],
        8: ['right_leg', [17]],
        9: ['left_shoe', [18]],
        10: ['right_shoe', [19]],
        11: ['socks', [8]],
        12: ['noise', [3, 11]]
    }

    def __init__(self, data_root_dir, img_H, img_W, phase, is_paired=True, is_sorted=False, transform_size=None,
                 transform_color=None, semantic_nc=None, use_preprocessed=False, **kwargs):
        super().__init__(data_root_dir, img_H, img_W, phase, is_paired, is_sorted, transform_size, transform_color,
                         **kwargs)
        self.transform = transforms.Compose([ \
            transforms.ToTensor(), \
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        self.semantic_nc = semantic_nc
        self.use_preprocessed = use_preprocessed

    def build_parse_agnostic(self, idx):
        # load parsing image
        parse_name = opj(self.drd, self.data_type, 'image-parse-v3', self.im_names[idx]).replace('.jpg', '.png')
        im_parse_pil_big = Image.open(parse_name)
        im_parse_pil = TF.resize(im_parse_pil_big,
                                 (int(self.img_H*self.resize_ratio_H), int(self.img_W*self.resize_ratio_W)),
                                 interpolation=InterpolationMode.NEAREST)
        parse = torch.from_numpy(np.array(im_parse_pil)[None]).long()
        im_parse = self.transform(im_parse_pil.convert('RGB'))


        parse_map = torch.FloatTensor(20, self.img_H, self.img_W).zero_()
        parse_map = parse_map.scatter_(0, parse, 1.0)
        new_parse_map = torch.FloatTensor(self.semantic_nc, self.img_H, self.img_W).zero_()

        for i in range(len(self.labels)):
            for label in self.labels[i][1]:
                new_parse_map[i] += parse_map[label]

        parse_onehot = torch.FloatTensor(1, self.img_H, self.img_W).zero_()
        for i in range(len(self.labels)):
            for label in self.labels[i][1]:
                parse_onehot[0] += parse_map[label] * i

        # load image-parse-agnostic
        image_parse_agnostic = Image.open(
            osp.join(parse_name.replace('image-parse-v3', 'image-parse-agnostic-v3.2')))
        image_parse_agnostic = transforms.Resize(self.img_W, interpolation=0)(image_parse_agnostic)
        parse_agnostic = torch.from_numpy(np.array(image_parse_agnostic)[None]).long()

        parse_agnostic_map = torch.FloatTensor(20, self.img_H, self.img_W).zero_()
        parse_agnostic_map = parse_agnostic_map.scatter_(0, parse_agnostic, 1.0)
        new_parse_agnostic_map = torch.FloatTensor(self.semantic_nc, self.img_H, self.img_W).zero_()
        for i in range(len(self.labels)):
            for label in self.labels[i][1]:
                new_parse_agnostic_map[i] += parse_agnostic_map[label]
        return new_parse_agnostic_map

    def __getitem__(self, idx):
        img_fn = self.im_names[idx]
        cloth_fn = self.c_names[self.pair_key][idx]
        agn = imread_for_albu(opj(self.drd, self.data_type, "agnostic-v3.2", self.im_names[idx]))
        agn_mask = imread_for_albu(
            opj(self.drd, self.data_type, "agnostic-mask", self.im_names[idx].replace(".jpg", "_mask.png")),
            is_mask=True)
        cloth = imread_for_albu(opj(self.drd, self.data_type, "cloth", self.c_names[self.pair_key][idx]))
        cloth_mask = imread_for_albu(opj(self.drd, self.data_type, "cloth-mask", self.c_names[self.pair_key][idx]),
                                     is_mask=True, cloth_mask_check=True)

        gt_cloth_warped_mask = imread_for_albu(
            opj(self.drd, self.data_type, "gt_cloth_warped_mask", self.im_names[idx]),
            is_mask=True
        ) if not self.is_test else np.zeros_like(agn_mask)
        hybvton_warped_cloth = imread_for_albu(
            opj(self.drd, self.data_type, "hybvton_warped_cloth_" + self.pair_key,
                self.im_names[idx].split(".")[0] + "_" + self.c_names[self.pair_key][idx].replace(".jpg", ".png")),
        )
        hybvton_warped_mask = imread_for_albu(
            opj(self.drd, self.data_type, "hybvton_warped_mask_" + self.pair_key,
                self.im_names[idx].split(".")[0] + "_" + self.c_names[self.pair_key][idx].replace(".jpg", ".png")),
            is_mask=True
        )

        image = imread_for_albu(opj(self.drd, self.data_type, "image", self.im_names[idx]))
        image_densepose = imread_for_albu(opj(self.drd, self.data_type, "image-densepose", self.im_names[idx]))
        image_densepose_hybvton = imread_for_albu(opj(
            self.drd, self.data_type, "image-densepose_hybvton", self.im_names[idx].replace(".jpg", ".png")))
        densepose_torso_mask = image_densepose_hybvton[:, :, 0] == DENSEPOSE_SEGM_RGB_TORSO[0]
        densepose_torso_mask = densepose_torso_mask.astype(np.uint8)[:,:,None]

        if self.transform_size is not None:
            transformed = self.transform_size(
                image=image,
                agn=agn,
                agn_mask=agn_mask,
                cloth=cloth,
                cloth_mask=cloth_mask,
                image_densepose=image_densepose,
                gt_cloth_warped_mask=gt_cloth_warped_mask,
                densepose_torso_mask=densepose_torso_mask,
            )
            image = transformed["image"]
            agn = transformed["agn"]
            agn_mask = transformed["agn_mask"]
            image_densepose = transformed["image_densepose"]
            gt_cloth_warped_mask = transformed["gt_cloth_warped_mask"]
            densepose_torso_mask = transformed["densepose_torso_mask"]

            cloth = transformed["cloth"]
            cloth_mask = transformed["cloth_mask"]

        # Hybvton: images and masks we add for refinement will have format c h w
        densepose_torso_mask = densepose_torso_mask.astype(np.float32).transpose(2, 0, 1)
        hybvton_warped_mask = (hybvton_warped_mask / 255 * agn_mask / 255)
        agn_mask_orig = 255 - agn_mask
        agn_orig = agn
        agn = agn * (1 - hybvton_warped_mask[:, :, None]) + hybvton_warped_cloth * hybvton_warped_mask[:, :, None]
        agn = agn.astype(np.uint8)
        agn_orig = agn_orig.astype(np.uint8)
        hybvton_warped_mask = (hybvton_warped_mask * 255).astype(np.uint8)
        agn_mask = np.clip(agn_mask - hybvton_warped_mask, 0, 255)

        if self.transform_hflip is not None:
            transformed = self.transform_hflip(
                image=image,
                agn=agn,
                agn_mask=agn_mask,
                cloth=cloth,
                cloth_mask=cloth_mask,
                image_densepose=image_densepose,
                gt_cloth_warped_mask=gt_cloth_warped_mask,
                hybvton_warped_cloth=hybvton_warped_cloth,
                hybvton_warped_mask=hybvton_warped_mask,
            )

            image = transformed["image"]
            agn = transformed["agn"]
            agn_mask = transformed["agn_mask"]
            image_densepose = transformed["image_densepose"]
            gt_cloth_warped_mask = transformed["gt_cloth_warped_mask"]
            hybvton_warped_cloth = transformed["hybvton_warped_cloth"]
            hybvton_warped_mask = transformed["hybvton_warped_mask"]

            cloth = transformed["cloth"]
            cloth_mask = transformed["cloth_mask"]

        if self.transform_crop_person is not None:
            transformed_image = self.transform_crop_person(
                image=image,
                agn=agn,
                agn_mask=agn_mask,
                image_densepose=image_densepose,
                gt_cloth_warped_mask=gt_cloth_warped_mask,
                hybvton_warped_cloth=hybvton_warped_cloth,
                hybvton_warped_mask=hybvton_warped_mask,
            )

            image = transformed_image["image"]
            agn = transformed_image["agn"]
            agn_mask = transformed_image["agn_mask"]
            image_densepose = transformed_image["image_densepose"]
            gt_cloth_warped_mask = transformed_image["gt_cloth_warped_mask"]
            hybvton_warped_cloth = transformed_image["hybvton_warped_cloth"]
            hybvton_warped_mask = transformed_image["hybvton_warped_mask"]

        if self.transform_crop_cloth is not None:
            transformed_cloth = self.transform_crop_cloth(
                image=cloth,
                cloth_mask=cloth_mask
            )

            cloth = transformed_cloth["image"]
            cloth_mask = transformed_cloth["cloth_mask"]

        agn_mask = 255 - agn_mask
        if self.transform_color is not None:
            transformed = self.transform_color(
                image=image,
                agn=agn,
                cloth=cloth,
            )

            image = transformed["image"]
            agn = transformed["agn"]
            cloth = transformed["cloth"]

            agn = agn * agn_mask[:, :, None].astype(np.float32) / 255.0 + 128 * (
                        1 - agn_mask[:, :, None].astype(np.float32) / 255.0)

        agn = norm_for_albu(agn)
        agn_orig = norm_for_albu(agn_orig)
        agn_mask_orig = norm_for_albu(agn_mask_orig, is_mask=True)
        agn_mask = norm_for_albu(agn_mask, is_mask=True)
        cloth = norm_for_albu(cloth)
        cloth_mask = norm_for_albu(cloth_mask, is_mask=True)
        image = norm_for_albu(image)
        image_densepose = norm_for_albu(image_densepose)
        gt_cloth_warped_mask = norm_for_albu(gt_cloth_warped_mask, is_mask=True)
        hybvton_warped_cloth = norm_for_albu(hybvton_warped_cloth)
        hybvton_warped_mask = norm_for_albu(hybvton_warped_mask, is_mask=True)

        # original warped cloth and mask
        warped_cloth_pil = Image.open(osp.join(self.drd, self.data_type, f'hybvton_warped_cloth_{self.pair_key}_orig',
                                               self.im_names[idx].split(".")[0] + "_" + self.c_names[self.pair_key][
                                                   idx].replace(".jpg", ".png")))
        warped_cloth_pil = TF.resize(warped_cloth_pil,
                                     (int(self.img_H*self.resize_ratio_H), int(self.img_W*self.resize_ratio_W)),
                                     interpolation=InterpolationMode.BICUBIC)
        warped_cloth_np = np.array(warped_cloth_pil)

        warped_mask_pil = Image.open(osp.join(self.drd, self.data_type, f'hybvton_warped_mask_{self.pair_key}_orig',
                                              self.im_names[idx].split(".")[0] + "_" + self.c_names[self.pair_key][
                                                  idx].replace(".jpg", ".png")))
        warped_mask_pil = TF.resize(warped_mask_pil,
                                    (int(self.img_H*self.resize_ratio_H), int(self.img_W*self.resize_ratio_W)),
                                    interpolation=InterpolationMode.NEAREST)
        warped_mask_np = np.array(warped_mask_pil)
        warped_mask_orig = torch.from_numpy(warped_mask_np >= 0.5).float()[None]
        warped_cloth_orig = self.transform(warped_cloth_np)

        parse_agnostic = self.build_parse_agnostic(idx)

        warped_cloth_processed = 0
        if self.use_preprocessed:
            warped_cloth_processed_pil = Image.open(
                osp.join(self.drd, self.data_type, f'hybvton_warped_cloth_{self.pair_key}_processed',
                         self.im_names[idx].split(".")[0] + "_" + self.c_names[self.pair_key][
                             idx].replace(".jpg", ".png")))
            warped_cloth_processed_pil = TF.resize(warped_cloth_processed_pil,
                                         (int(self.img_H * self.resize_ratio_H), int(self.img_W * self.resize_ratio_W)),
                                         interpolation=InterpolationMode.BICUBIC)
            warped_cloth_processed = self.transform(warped_cloth_processed_pil)


        return dict(
            agn=agn,
            agn_orig=agn_orig,
            agn_mask=agn_mask,
            cloth=cloth,
            cloth_mask=cloth_mask,
            image=image,
            image_densepose=image_densepose,
            gt_cloth_warped_mask=gt_cloth_warped_mask,
            txt="",
            img_fn=img_fn,
            cloth_fn=cloth_fn,
            hybvton_warped_cloth=hybvton_warped_cloth,
            hybvton_warped_mask=hybvton_warped_mask,
            agn_mask_orig=agn_mask_orig,
            warped_mask_orig=warped_mask_orig,
            warped_cloth_orig=warped_cloth_orig,
            parse_agnostic=parse_agnostic,
            densepose_torso_mask=densepose_torso_mask,
            warped_cloth_processed=warped_cloth_processed,
        )
