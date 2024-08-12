"""
"Copyright (C) 2021 Samsung Electronics Co. LTD

This software is a property of Samsung Electronics.
No part of this software, either material or conceptual may be copied or distributed, transmitted,
transcribed, stored in a retrieval system, or translated into any human or computer language in any form by any means,
electronic, mechanical, manual or otherwise, or disclosed
to third parties without the express written permission of Samsung Electronics.
(Use of the Software is restricted to non-commercial, personal or academic, research purpose only)"
"""

"""
Modified from NerfAcc (https://github.com/KAIR-BAIR/nerfacc)
Copyright (c) 2022 Ruilong Li, UC Berkeley.
"""

import collections
import json
import os
from tqdm import tqdm
import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn.functional as F

from .utils import Rays


def circle(radius=3.5, h=0.0, axis='z', t0=0, r=1):
    if axis == 'z':
        return lambda t: [radius * np.cos(r * t + t0), radius * np.sin(r * t + t0), h]
    elif axis == 'y':
        return lambda t: [radius * np.cos(r * t + t0), h, radius * np.sin(r * t + t0)]
    else:
        return lambda t: [h, radius * np.cos(r * t + t0), radius * np.sin(r * t + t0)]


def _load_renderings(root_fp: str, subject_id: str, split: str):
    """Load images from disk."""
    if not root_fp.startswith("/"):
        # allow relative path. e.g., "./data/nerf_synthetic/"
        root_fp = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..",
            "..",
            root_fp,
        )

    data_dir = os.path.join(root_fp, subject_id)

    img_files = sorted(os.listdir(os.path.join(data_dir, "rgb")))
    pose_files = sorted(os.listdir(os.path.join(data_dir, "pose")))

    if split == "train":
        img_files = [os.path.join(data_dir, "rgb", img_file) for img_file in img_files if img_file.startswith("0_")]
        pose_files = [os.path.join(data_dir, "pose", pose_file) for pose_file in pose_files if pose_file.startswith("0_")]
    elif split == "test":
        img_files = [os.path.join(data_dir, "rgb", img_file) for img_file in img_files if img_file.startswith("1_")]
        pose_files = [os.path.join(data_dir, "pose", pose_file) for pose_file in pose_files if pose_file.startswith("1_")]
    elif split == "render" or split == 'val':
        img_files = [os.path.join(data_dir, "rgb", img_file) for img_file in img_files if img_file.startswith("1_")]
        pose_files = [os.path.join(data_dir, "pose", pose_file) for pose_file in pose_files if pose_file.startswith("1_")]
    images = []
    camtoworlds = []
    print('dataset loading...')
    for i in tqdm(range(len(img_files))):
        images.append(imageio.imread(img_files[i]))
        camtoworlds.append(np.loadtxt(pose_files[i]))

    images = np.stack(images, axis=0)
    if split == "render":
        images = images[0:1].repeat(len(pose_files), axis=0)
        camtoworlds = []
        for i in range(len(pose_files)):
            camtoworlds.append(np.loadtxt(pose_files[i]))
    
    camtoworlds = np.stack(camtoworlds, axis=0)

    h, w = images.shape[1:3]
    with open(os.path.join(data_dir, "intrinsics.txt")) as f:
        focal = float(f.readline().split()[0])

    aabb = np.loadtxt(os.path.join(data_dir, "bbox.txt"))[:6]

    return images, camtoworlds, focal


class SubjectLoader(torch.utils.data.Dataset):
    """Single subject data loader for training and evaluation."""

    SPLITS = ["train", "val", "trainval", "test", "render"]
    SUBJECT_IDS = [
        "Barn", 
        "Caterpillar", 
        "Family", 
        "Ignatius", 
        "Truck",
    ]

    WIDTH, HEIGHT = 1920, 1080
    NEAR, FAR = 0.01, 6.0
    OPENGL_CAMERA = False

    def __init__(
        self,
        subject_id: str,
        root_fp: str,
        split: str,
        color_bkgd_aug: str = "black",
        num_rays: int = None,
        near: float = None,
        far: float = None,
        batch_over_images: bool = True,
    ):
        super().__init__()
        assert split in self.SPLITS, "%s" % split
        assert subject_id in self.SUBJECT_IDS, "%s" % subject_id
        assert color_bkgd_aug in ["white", "black", "random"]
        self.split = split
        self.num_rays = num_rays
        self.near = self.NEAR if near is None else near
        self.far = self.FAR if far is None else far
        self.training = (num_rays is not None) and (
            split in ["train", "trainval"]
        )
        self.color_bkgd_aug = color_bkgd_aug
        self.batch_over_images = batch_over_images
        if split == "trainval":
            _images_train, _camtoworlds_train, _focal_train = _load_renderings(
                root_fp, subject_id, "train"
            )
            _images_val, _camtoworlds_val, _focal_val = _load_renderings(
                root_fp, subject_id, "val"
            )
            self.images = np.concatenate([_images_train, _images_val])
            self.camtoworlds = np.concatenate(
                [_camtoworlds_train, _camtoworlds_val]
            )
            self.focal = _focal_train
        else:
            self.images, self.camtoworlds, self.focal = _load_renderings(
                root_fp, subject_id, split
            )
        self.images = torch.from_numpy(self.images).to(torch.uint8)
        self.camtoworlds = torch.from_numpy(self.camtoworlds).to(torch.float32)

        root_fp = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..",
            "..",
            root_fp,
        )
        data_dir = os.path.join(root_fp, subject_id)

        if subject_id == "Ignatius":
            print(subject_id)
            self.K = torch.tensor(
                [
                    [self.focal, 0, self.WIDTH / 2.0],
                    [0, self.focal, self.HEIGHT / 2.0],
                    [0, 0, 1],
                ],
                dtype=torch.float32,
            )  # (3, 3)
        else:
            self.K = torch.tensor(
                np.loadtxt(
                    os.path.join(data_dir, 'intrinsics.txt')
                )[:3, :3], dtype=torch.float32,
            )  # (3, 3)

        assert self.images.shape[1:3] == (self.HEIGHT, self.WIDTH)
