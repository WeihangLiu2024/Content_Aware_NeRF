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

from .utils import get_rays, create_dodecahedron_cameras, linear_to_srgb

def nerf_matrix_to_ngp(pose, scale=0.33, offset=[0, 0, 0]):
    pose[:3, 3] = pose[:3, 3] * scale + np.array(offset)
    pose = pose.astype(np.float32)
    return pose

# def nerf_matrix_to_ngp(pose, scale=0.33, offset=[0, 0, 0]):
#     # for the fox dataset, 0.33 scales camera radius to ~ 2
#     new_pose = np.array([
#         [pose[1, 0], -pose[1, 1], -pose[1, 2], pose[1, 3] * scale + offset[0]],
#         [pose[2, 0], -pose[2, 1], -pose[2, 2], pose[2, 3] * scale + offset[1]],
#         [pose[0, 0], -pose[0, 1], -pose[0, 2], pose[0, 3] * scale + offset[2]],
#         [0, 0, 0, 1],
#     ], dtype=np.float32)
#     return new_pose


def visualize_poses(poses, size=0.1, bound=1):
    # poses: [B, 4, 4]

    axes = trimesh.creation.axis(axis_length=4)
    box = trimesh.primitives.Box(extents=[2*bound]*3).as_outline()
    box.colors = np.array([[128, 128, 128]] * len(box.entities))
    objects = [axes, box]

    if bound > 1:
        unit_box = trimesh.primitives.Box(extents=[2]*3).as_outline()
        unit_box.colors = np.array([[128, 128, 128]] * len(unit_box.entities))
        objects.append(unit_box)

    for pose in poses:
        # a camera is visualized with 8 line segments.
        pos = pose[:3, 3]
        a = pos + size * pose[:3, 0] + size * pose[:3, 1] - size * pose[:3, 2]
        b = pos - size * pose[:3, 0] + size * pose[:3, 1] - size * pose[:3, 2]
        c = pos - size * pose[:3, 0] - size * pose[:3, 1] - size * pose[:3, 2]
        d = pos + size * pose[:3, 0] - size * pose[:3, 1] - size * pose[:3, 2]

        dir = (a + b + c + d) / 4 - pos
        dir = dir / (np.linalg.norm(dir) + 1e-8)
        o = pos + dir * 3

        segs = np.array([[pos, a], [pos, b], [pos, c], [pos, d], [a, b], [b, c], [c, d], [d, a], [pos, o]])
        segs = trimesh.load_path(segs)
        objects.append(segs)

    trimesh.Scene(objects).show()


def rand_poses(size, device, radius=1, theta_range=[np.pi/3, 2*np.pi/3], phi_range=[0, 2*np.pi]):
    ''' generate random poses from an orbit camera
    Args:
        size: batch size of generated poses.
        device: where to allocate the output.
        radius: camera radius
        theta_range: [min, max], should be in [0, \pi]
        phi_range: [min, max], should be in [0, 2\pi]
    Return:
        poses: [size, 4, 4]
    '''
    
    def normalize(vectors):
        return vectors / (torch.norm(vectors, dim=-1, keepdim=True) + 1e-10)

    thetas = torch.rand(size, device=device) * (theta_range[1] - theta_range[0]) + theta_range[0]
    phis = torch.rand(size, device=device) * (phi_range[1] - phi_range[0]) + phi_range[0]

    centers = torch.stack([
        radius * torch.sin(thetas) * torch.sin(phis),
        radius * torch.cos(thetas),
        radius * torch.sin(thetas) * torch.cos(phis),
    ], dim=-1) # [B, 3]

    # lookat
    forward_vector = - normalize(centers)
    up_vector = torch.FloatTensor([0, -1, 0]).to(device).unsqueeze(0).repeat(size, 1) # confused at the coordinate system...
    right_vector = normalize(torch.cross(forward_vector, up_vector, dim=-1))
    up_vector = normalize(torch.cross(right_vector, forward_vector, dim=-1))

    poses = torch.eye(4, dtype=torch.float, device=device).unsqueeze(0).repeat(size, 1, 1)
    poses[:, :3, :3] = torch.stack((right_vector, up_vector, forward_vector), dim=-1)
    poses[:, :3, 3] = centers

    return poses


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

        if self.scale == -1:
            print(f'[WARN] --data_format nerf cannot auto-choose --scale, use 1 as default.')
            self.scale = 1
            
        self.training = self.type in ['train', 'all', 'trainval']

        # auto-detect transforms.json and split mode.
        if os.path.exists(os.path.join(self.root_path, 'transforms.json')):
            self.mode = 'colmap' # manually split, use view-interpolation for test. (merely use [only for fox], waiting to delete)
        elif os.path.exists(os.path.join(self.root_path, 'transforms_train.json')):
            self.mode = 'blender' # provided split
        elif os.path.exists(os.path.join(self.root_path, '00000.json')):
            self.mode = 'rtmv'  # manually split with "self.train_spilt"
            self.train_spilt = 0.8  # ratio that training set comprising of
        else:
            raise NotImplementedError(f'[NeRFDataset] Cannot find transforms*.json under {self.root_path}')

        if self.mode == 'rtmv':
            import pyexr
            # SURELY TO DEFINE: self.H; self.W; self.poses; self.images; self.intrinsics
            # UNCERTAIN: self.error_map; self.radius
            self.poses = []
            self.images = []
            # ===== load all filenames =====
            image_filenames = glob.glob(os.path.join(self.root_path, '*.exr'))
            depth_filenames = glob.glob(os.path.join(self.root_path, '*.depth.exr'))
            seg_filenames = glob.glob(os.path.join(self.root_path, '*.seg.exr'))
            image_filenames = [f for f in image_filenames if f not in depth_filenames]
            image_filenames = [f for f in image_filenames if f not in seg_filenames]
            transform_paths = glob.glob(os.path.join(self.root_path, '*.json'))

            camera_filenames = [f for f in transform_paths]
            camera_filenames.sort()
            image_filenames.sort()
            assert len(image_filenames) == len(camera_filenames)
            sum_num = len(image_filenames)
            train_num = round(sum_num * self.train_spilt)

            # all json files share one set of H, W, intrinsics
            with open(os.path.join(camera_filenames[0]), 'r') as fp:
                meta = json.load(fp)
                camera_data = meta['camera_data']
                self.H = camera_data['height']
                self.W = camera_data['width']
                cx = camera_data['intrinsics']['cx']
                cy = camera_data['intrinsics']['cy']
                fl_x = camera_data['intrinsics']['fx']
                fl_y = camera_data['intrinsics']['fy']
                self.intrinsics = np.array([fl_x, fl_y, cx, cy])

            if self.type == 'train':
                image_filenames = image_filenames[:train_num]
                camera_filenames = camera_filenames[:train_num]
            else:  # val & test
                image_filenames = image_filenames[train_num:]
                camera_filenames = camera_filenames[train_num:]

            # load images and poses
            num_iterations = len(image_filenames)
            progress_bar = tqdm.tqdm(total=num_iterations, desc=f'Loading {type} data')
            for image_filename, camera_filename in zip(image_filenames, camera_filenames):
                image_path = os.path.join(image_filename)
                camera_path = os.path.join(camera_filename)

                f = pyexr.open(image_path)
                image = f.get("default")
                image = torch.tensor(image)
                image[:, :, :3] = linear_to_srgb(image[:, :, :3])
                self.images.append(image)

                with open(os.path.join(camera_path), 'r') as fp:
                    meta = json.load(fp)
                    camera_data = meta['camera_data']
                    pose = (np.array(camera_data['cam2world'], dtype=np.float32)).T
                    # line below is very important!!!
                    pose = nerf_matrix_to_ngp(pose, scale=self.scale, offset=self.offset)
                    self.poses.append(pose)
                progress_bar.update(1)
            progress_bar.close()
        else:
            # load nerf-compatible format data.
            if self.mode == 'colmap':
                with open(os.path.join(self.root_path, 'transforms.json'), 'r') as f:
                    transform = json.load(f)
            elif self.mode == 'blender':
                # load all splits (train/valid/test), this is what instant-ngp in fact does...
                if type == 'all':
                    transform_paths = glob.glob(os.path.join(self.root_path, '*.json'))
                    transform = None
                    for transform_path in transform_paths:
                        with open(transform_path, 'r') as f:
                            tmp_transform = json.load(f)
                            if transform is None:
                                transform = tmp_transform
                            else:
                                transform['frames'].extend(tmp_transform['frames'])
                # load train and val split
                elif type == 'trainval':
                    with open(os.path.join(self.root_path, f'transforms_train.json'), 'r') as f:
                        transform = json.load(f)
                    with open(os.path.join(self.root_path, f'transforms_val.json'), 'r') as f:
                        transform_val = json.load(f)
                    transform['frames'].extend(transform_val['frames'])
                # only load one specified split
                else:
                    with open(os.path.join(self.root_path, f'transforms_{type}.json'), 'r') as f:
                        transform = json.load(f)

            else:
                raise NotImplementedError(f'unknown dataset mode: {self.mode}')

            # load image size
            if 'h' in transform and 'w' in transform:
                self.H = int(transform['h']) // self.downscale
                self.W = int(transform['w']) // self.downscale
            else:
                # we have to actually read an image to get H and W later.
                self.H = self.W = None

            # read images
            frames = np.array(transform["frames"])

            # for colmap, manually interpolate a test set.
            if self.mode == 'colmap' and type == 'test':

                # choose two random poses, and interpolate between.
                f0, f1 = np.random.choice(frames, 2, replace=False)
                pose0 = nerf_matrix_to_ngp(np.array(f0['transform_matrix'], dtype=np.float32), scale=self.scale, offset=self.offset) # [4, 4]
                pose1 = nerf_matrix_to_ngp(np.array(f1['transform_matrix'], dtype=np.float32), scale=self.scale, offset=self.offset) # [4, 4]
                rots = Rotation.from_matrix(np.stack([pose0[:3, :3], pose1[:3, :3]]))
                slerp = Slerp([0, 1], rots)

                self.poses = []
                self.images = None
                for i in range(n_test + 1):
                    ratio = np.sin(((i / n_test) - 0.5) * np.pi) * 0.5 + 0.5
                    pose = np.eye(4, dtype=np.float32)
                    pose[:3, :3] = slerp(ratio).as_matrix()
                    pose[:3, 3] = (1 - ratio) * pose0[:3, 3] + ratio * pose1[:3, 3]
                    self.poses.append(pose)

            else:
                # for colmap, manually split a valid set (the first frame).
                if self.mode == 'colmap':
                    if type == 'train':
                        frames = frames[1:]
                    elif type == 'val':
                        frames = frames[:1]
                    # else 'all' or 'trainval' : use all frames

                self.poses = []
                self.images = []
                for f in tqdm.tqdm(frames, desc=f'Loading {type} data'):
                    image_path = os.path.join(self.root_path, f['file_path'])
                    if self.mode == 'blender' and '.' not in os.path.basename(image_path):
                        image_path += '.png' # so silly...

                    # there are non-exist paths in fox...
                    if not os.path.exists(image_path):
                        print(f'[WARN] {image_path} not exists!')
                        # only transform matrix is OK for test.
                        pose = np.array(f['transform_matrix'], dtype=np.float32)  # [4, 4]
                        pose = nerf_matrix_to_ngp(pose, scale=self.scale, offset=self.offset)
                        self.poses.append(pose)
                        continue

                    pose = np.array(f['transform_matrix'], dtype=np.float32) # [4, 4]
                    pose = nerf_matrix_to_ngp(pose, scale=self.scale, offset=self.offset)

                    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED) # [H, W, 3] o [H, W, 4]
                    if self.H is None or self.W is None:
                        self.H = image.shape[0] // self.downscale
                        self.W = image.shape[1] // self.downscale

                    # add support for the alpha channel as a mask.
                    if image.shape[-1] == 3:
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    else:
                        image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)

                    if image.shape[0] != self.H or image.shape[1] != self.W:
                        image = cv2.resize(image, (self.W, self.H), interpolation=cv2.INTER_AREA)

                    self.poses.append(pose)
                    self.images.append(image)
            # load intrinsics
            if 'fl_x' in transform or 'fl_y' in transform:
                fl_x = (transform['fl_x'] if 'fl_x' in transform else transform['fl_y']) / self.downscale
                fl_y = (transform['fl_y'] if 'fl_y' in transform else transform['fl_x']) / self.downscale
            elif 'camera_angle_x' in transform or 'camera_angle_y' in transform:
                # blender, assert in radians. already downscaled since we use H/W
                fl_x = self.W / (2 * np.tan(transform['camera_angle_x'] / 2)) if 'camera_angle_x' in transform else None
                fl_y = self.H / (2 * np.tan(transform['camera_angle_y'] / 2)) if 'camera_angle_y' in transform else None
                if fl_x is None: fl_x = fl_y
                if fl_y is None: fl_y = fl_x
            else:
                raise RuntimeError('Failed to load focal length, please check the transforms.json!')

            cx = (transform['cx'] / self.downscale) if 'cx' in transform else (self.W / 2.0)
            cy = (transform['cy'] / self.downscale) if 'cy' in transform else (self.H / 2.0)

            self.intrinsics = np.array([fl_x, fl_y, cx, cy])

        self.poses = torch.from_numpy(np.stack(self.poses, axis=0)) # [N, 4, 4]
        if self.images is not None and len(self.images):
            # self.images = torch.from_numpy(np.stack(self.images, axis=0).astype(np.uint8)) # [N, H, W, C]
            self.images = torch.from_numpy(np.stack(self.images, axis=0))  # [N, H, W, C]
        
        # calculate mean radius of all camera poses
        self.radius = self.poses[:, :3, 3].norm(dim=-1).mean(0).item()
        #print(f'[INFO] dataset camera poses: radius = {self.radius:.4f}, bound = {self.bound}')

        # [debug] uncomment to view all training poses.
        if self.opt.vis_pose:
            visualize_poses(self.poses.numpy(), bound=self.opt.bound)

        # perspective projection matrix
        self.near = self.opt.min_near
        self.far = 1000 # infinite
        y = self.H / (2.0 * fl_y)
        aspect = self.W / self.H
        self.projection = np.array([[1/(y*aspect), 0, 0, 0], 
                                    [0, -1/y, 0, 0],
                                    [0, 0, -(self.far+self.near)/(self.far-self.near), -(2*self.far*self.near)/(self.far-self.near)],
                                    [0, 0, -1, 0]], dtype=np.float32)

        self.projection = torch.from_numpy(self.projection)
        self.mvps = self.projection.unsqueeze(0) @ torch.inverse(self.poses)
    
        # tmp: dodecahedron_cameras for mesh visibility test
        dodecahedron_poses = create_dodecahedron_cameras()
        # visualize_poses(dodecahedron_poses, bound=self.opt.bound, points=self.pts3d)
        self.dodecahedron_poses = torch.from_numpy(dodecahedron_poses.astype(np.float32)) # [N, 4, 4]
        self.dodecahedron_mvps = self.projection.unsqueeze(0) @ torch.inverse(self.dodecahedron_poses)

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
        rays = get_rays(poses, self.intrinsics, self.H, self.W, num_rays)

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
