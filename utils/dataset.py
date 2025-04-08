from PIL import Image
import math
import torch
import numpy as np
import json


def normalize_image(image):
    return (image - 0.5) * 2.0


def denormalize_image(image):
    return (image + 1.0) / 2.0


def add_margin(image: Image, top, right, bottom, left, color):
    width, height = image.size
    new_width = width + right + left
    new_height = height + top + bottom
    result = Image.new(image.mode, (new_width, new_height), color)
    result.paste(image, (left, top))
    return result


def resize_image(image: Image, shape, fill, resample=Image.Resampling.BICUBIC):
    # image: (C, H, W)
    # shape: (H, W)
    # fill: (R, G, B) or int
    # returns: (C, H, W)
    #
    # resize the image to the given shape while preserving the aspect ratio
    if isinstance(fill, int):
        fill = (fill, fill, fill)
    orig_h, orig_w = image.size[::-1]
    if isinstance(shape, int):
        shape = (shape, shape)
    new_h, new_w = shape
    # in case both has the same aspect ratio
    if orig_h / orig_w == new_h / new_w:
        result = image.resize(shape[::-1], resample=resample)
    elif orig_h / orig_w > new_h / new_w:
        tmp_w = int(new_h / orig_h * orig_w)
        image = image.resize([tmp_w, new_h], resample=resample)
        lpad = (new_w - tmp_w) // 2
        rpad = lpad + (new_w - tmp_w) % 2
        result = add_margin(image, 0, rpad, 0, lpad, fill)
    else:
        tmp_h = int(new_w / orig_w * orig_h)
        image = image.resize([new_w, tmp_h], resample=resample)
        tpad = (new_h - tmp_h) // 2
        bpad = tpad + (new_h - tmp_h) % 2
        result = add_margin(image, tpad, 0, bpad, 0, fill)
    return result

def resize_pose(pose, shape, image_shape_orig):
    # pose: (pose_keypoints, 2)
    orig_h, orig_w = image_shape_orig
    new_h, new_w = shape
    if orig_h / orig_w == new_h / new_w:
        ratio = new_w / orig_w
        return pose * ratio
    elif orig_h / orig_w > new_h / new_w:
        ratio = new_h / orig_h
        tmp_w = int(new_h / orig_h * orig_w)
        pad = (new_w - tmp_w) // 2
        pose = pose * ratio
        pose[:, 0] = pose[:, 0] + pad
    else:
        ratio = new_w / orig_w
        tmp_h = int(new_w / orig_w * orig_h)
        pad = (new_h - tmp_h) // 2
        pose = pose * ratio
        pose[:, 1] = pose[:, 1] + pad
    return pose


def normalize_pose_keypoints(self, pose, image_shape_orig):
    shape = self.resize_shape if self.resize_shape is not None else image_shape_orig
    pose[:, 0] = pose[:, 0] / shape[1]
    pose[:, 1] = pose[:, 1] / shape[0]
    return pose


def rotate_pose(self, pose, degree):
    pose[:, 0] = pose[:, 0] - 0.5
    pose[:, 1] = -(pose[:, 1] - 0.5)
    radius = math.radians(degree)
    rotation_matrix = torch.tensor([
        [math.cos(radius), math.sin(radius)],
        [-math.sin(radius), math.cos(radius)]
    ])
    pose = torch.matmul(pose, rotation_matrix)
    pose[:, 0] = pose[:, 0] + 0.5
    pose[:, 1] = -pose[:, 1] + 0.5
    return pose


def flip_pose(pose, image_shape_orig):
    # applied order 1
    pose[:, 0] = image_shape_orig[1] - pose[:, 0]
    return pose


def extract_openpose(pose_path):
    with open(pose_path, 'r') as f:
        pose_dict = json.load(f)
    pose = np.array(pose_dict['people'][0]['pose_keypoints_2d']).reshape(-1, 3)[:, :2]
    pose = torch.from_numpy(pose).float()
    return pose


def denormalized_to_uint8_np(image, positive=False):
    if not positive:
        image = (image + 1) / 2
    image = (image * 255 + 0.5).astype(np.uint8)
    return image
