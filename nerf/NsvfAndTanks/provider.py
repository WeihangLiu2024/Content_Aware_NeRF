"""
Simply transform the NSVF and TanksAndTemple data loaders in the BiRF(https://github.com/SAITPublic/BiRF)
into usable data loaders for this project
"""

import os
import cv2
import glob
import json
import tqdm
import numpy as np
from scipy.spatial.transform import Slerp, Rotation

import trimesh

import torch
from torch.utils.data import DataLoader

from ..utils import get_rays, create_dodecahedron_cameras, linear_to_srgb, get_rays2


def convert(c2w:torch.tensor):
    transform_matrix = torch.tensor([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ], device=c2w.device, dtype=torch.float32)
    c2w = torch.mm(transform_matrix, c2w)
    return c2w


def load_dataset(
    scene: str,
    data_root_fp: str,
    split: str,
    num_rays,
    dataset_kwargs,
    device: str,
):
    if scene in ["chair", "drums", "ficus", "hotdog", "lego", "materials", "mic", "ship"]:
        from nerf.NsvfAndTanks.nerf_synthetic import SubjectLoader
        data_root_fp = 'data/nerf_synthetic/'
    elif scene in ["Bike", "Lifestyle", "Palace", "Robot", "Spaceship", "Steamtrain", "Toad", "Wineholder"]:
        from nerf.NsvfAndTanks.nsvf import SubjectLoader
        data_root_fp = 'data/Synthetic_NSVF/'
    elif scene in ["Barn", "Caterpillar", "Family", "Ignatius", "Truck"]:
        from nerf.NsvfAndTanks.tanksandtemple import SubjectLoader
        data_root_fp = 'data/TanksAndTemple/'

    dataset = SubjectLoader(
        subject_id=scene,
        root_fp=data_root_fp,
        split=split,
        num_rays=num_rays,
        **dataset_kwargs,
    )

    dataset.images = dataset.images.to(device)
    dataset.camtoworlds = dataset.camtoworlds.to(device)
    dataset.K = dataset.K.to(device)

    return dataset


class NeRFDataset:
    def __init__(self, opt, device, type='train', n_test=10):
        super().__init__()
        
        self.opt = opt
        self.device = device
        self.type = type # train, val, test
        self.downscale = opt.downscale
        self.root_path = opt.path
        self.preload = opt.preload # preload data into GPU
        self.scale = opt.scale # camera radius scale to make sure camera are inside the bounding box.
        self.offset = opt.offset # camera offset
        self.bound = opt.bound # bounding box half length, also used as the radius to random sample poses.
        self.fp16 = opt.fp16 # if preload, load into fp16.
        self.mode = 'blender'

        if self.scale == -1:
            self.scale = 1
            
        self.training = self.type in ['train', 'all', 'trainval']
        # data_loader frome BiRF 
        data_root_fp, scene = os.path.split(self.opt.path)
        train_dataset_kwargs = {}
        render_n_samples = self.opt.num_rays
        loader = load_dataset(
                scene=scene,
                data_root_fp=data_root_fp,
                split=self.type,
                num_rays=render_n_samples,
                dataset_kwargs=train_dataset_kwargs,
                device=self.device,
            )
        self.aabb = loader.aabb
        self.images = loader.images
        self.poses = loader.camtoworlds
        if self.opt.data_format == 'nerf':
            scale = 0.8
        elif self.opt.data_format == 'nsvf':
            scale = 1.0
        elif self.opt.data_format == 'tank':
            scale = 1.0
        else:
            scale = 1.0
        for i in range(self.poses.shape[0]):
            pose = self.poses[i]
            pose[:3, 3] = pose[:3, 3] * scale + torch.tensor([0, 0, 0]).to('cuda')
            pose = convert(pose)
            self.poses[i] = pose
        self.focal = loader.focal
        self.H = loader.HEIGHT
        self.W = loader.WIDTH
        self.K = loader.K
        self.OPENGL_CAMERA = loader.OPENGL_CAMERA
        cx = self.W / 2.0
        cy = self.H / 2.0
        fl_x = self.focal
        fl_y = self.focal
        self.intrinsics = np.array([fl_x, fl_y, cx, cy])
        self.radius = self.poses[:, :3, 3].norm(dim=-1).mean(0).item()
        self.near = self.opt.min_near
        self.far = 1000 # infinite
        # self.near = loader.NEAR
        # self.far = loader.FAR
        y = self.H / (2.0 * fl_y)
        aspect = self.W / self.H
        self.projection = np.array([[1/(y*aspect), 0, 0, 0], 
                                    [0, -1/y, 0, 0],
                                    [0, 0, -(self.far+self.near)/(self.far-self.near), -(2*self.far*self.near)/(self.far-self.near)],
                                    [0, 0, -1, 0]], dtype=np.float32)

        self.projection = torch.from_numpy(self.projection).to(self.device)
        self.mvps = self.projection.unsqueeze(0) @ torch.inverse(self.poses)
        
        # tmp: dodecahedron_cameras for mesh visibility test
        dodecahedron_poses = create_dodecahedron_cameras()
        # visualize_poses(dodecahedron_poses, bound=self.opt.bound, points=self.pts3d)
        self.dodecahedron_poses = torch.from_numpy(dodecahedron_poses.astype(np.float32)) # [N, 4, 4]
        self.dodecahedron_mvps = self.projection.unsqueeze(0) @ torch.inverse(self.dodecahedron_poses).to(self.device)
        # TODO
        # 现在的情况是无论参数是什么，都会preload
        if self.preload:
            self.poses = self.poses.to(self.device)
            if self.images is not None and len(self.images):
                self.images = self.images.to(self.device)
            self.projection = self.projection.to(self.device)
            self.mvps = self.mvps.to(self.device)


    def collate(self, index):
        B = len(index) # a list of length 1
        results = {'H': self.H, 'W': self.W}
        if self.training:
            # randomly sample over images too
            num_rays = self.opt.num_rays
            if self.opt.random_image_batch:
                index = torch.randint(0, len(self.poses), size=(num_rays,), device=self.device)
        else:
            num_rays = -1
        poses = self.poses[index].to(self.device) # [N, 4, 4]
        # images = self.images[index].to(self.device)  # [B, H, W, 3/4]
        # rays = get_rays(poses, self.intrinsics, self.H, self.W, num_rays, image=images)
        # rays = get_rays(poses, self.intrinsics, self.H, self.W, num_rays)
        rays = get_rays2(poses, self.intrinsics, self.H, self.W, self.K, self.OPENGL_CAMERA, num_rays)
        results['rays_o'] = rays['rays_o']
        results['rays_d'] = rays['rays_d']
        results['index'] = index
        mvp = self.mvps[index].to(self.device)
        results['mvp'] = mvp
        if self.images is not None and len(self.images):
            if self.mode == 'rtmv':
                scalar = 1
            else:
                scalar = 255
            if self.training:
                images = self.images[index, rays['j'], rays['i']].float().to(self.device) / scalar # [N, 3/4]
            else:
                images = self.images[index].squeeze(0).float().to(self.device) / scalar # [H, W, 3/4]
            if self.training:
                C = self.images.shape[-1]
                images = images.view(-1, C)
            results['images'] = images
        return results


    def dataloader(self):
        size = len(self.poses)
        loader = DataLoader(list(range(size)), batch_size=1, collate_fn=self.collate, shuffle=self.training,
                            num_workers=0)
        loader._data = self # an ugly fix... we need to access error_map & poses in trainer.
        loader.has_gt = self.images is not None
        return loader
