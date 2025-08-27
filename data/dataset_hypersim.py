from torch.utils.data import Dataset
import torch
import os
import glob
import struct
import numpy as np
import cv2
import h5py

EPS = 1e-6


def compute_shading(img, diffuse):
    shading = np.ones_like(diffuse) * EPS
    # shading = torch.ones_like(diffuse) * -1
    diffuse_nozero_mask = diffuse > EPS
    shading[diffuse_nozero_mask] = img[diffuse_nozero_mask] / diffuse[diffuse_nozero_mask]

    shading = shading.clip(EPS, 1e3)

    # shading = img / diffuse # It will lead to super large values
    return shading


class HypersimDataset(Dataset):
    def __init__(self, root_dir, mode='train', transforms=None, color_transforms=None, to_controlnet_input=None, invalid_fill_value=0):
        self.root_dir = root_dir
        self.mode = mode
        self.transforms = transforms
        self.color_transforms = color_transforms
        self.to_controlnet_input = to_controlnet_input
        self.invalid_fill_value = invalid_fill_value

        # Get scene names
        scene_names = glob.glob(os.path.join(self.root_dir, 'ai_*'))
        scene_names = [os.path.basename(p) for p in scene_names]
        if self.mode == 'train':
            scene_names = scene_names
            # scene_names = scene_names[:int(len(scene_names)*0.99)]
        elif self.mode == 'val':
            scene_names = scene_names[int(len(scene_names) * 0.99):]

        img_paths = glob.glob(os.path.join(self.root_dir, 'ai_*', 'images', '*final_preview', '*tonemap.jpg'))
        # Filter out images that are not in the scene_names
        img_paths = [p for p in img_paths if p.split(os.path.sep)[-4] in scene_names]
        img_paths = sorted(img_paths)
        self.directory_files = {
            'image': img_paths,
            'depth': [p.replace('final_preview', 'geometry_hdf5').replace('tonemap.jpg', 'depth_meters.hdf5') for p in img_paths],
            'normal': [p.replace('final_preview', 'geometry_preview').replace('tonemap.jpg', 'normal_cam.png') for p in img_paths],
            'diffuse': [p.replace('final_preview', 'final_hdf5').replace('tonemap.jpg', 'diffuse_reflectance.hdf5') for p in img_paths],
        }

        lack_list = []
        # Check if file exists
        for k, v in self.directory_files.items():
            assert len(v) > 0, f'{k} is empty'
            for i, p in enumerate(v):
                if not os.path.isfile(p):
                    print(f'{p} does not exist, deprecating...')
                    if i not in lack_list:
                        lack_list.append(i)

        for k, v in self.directory_files.items():
            self.directory_files[k] = [v[i] for i in range(len(v)) if i not in lack_list]
            for p in self.directory_files[k]:
                assert os.path.isfile(p), f'{p} does not exist'

    def __len__(self):
        return len(self.directory_files['image'])

    def __getitem__(self, idx):
        image_path = self.directory_files['image'][idx]
        depth_path = self.directory_files['depth'][idx]
        normal_path = self.directory_files['normal'][idx]
        diffuse_path = self.directory_files['diffuse'][idx]

        # This image is linear
        image = load_image(image_path, isGamma=False)

        # Load the distance from the camera to the pixel, not planar depth!
        cam_distance = load_hdf5(depth_path)
        cam_distance = np.expand_dims(cam_distance, axis=2)

        # Handle invalid values in depth maps (NaNs), NaN is actually the inf far away
        cam_distance = np.nan_to_num(cam_distance, nan=self.invalid_fill_value)
        # assert not np.isnan(depth).any(), f'{depth_path} has nan, it has {np.count_nonzero(np.isnan(depth))} nan values'

        # Convert camera distance to depth
        int_width = cam_distance.shape[1]
        int_height = cam_distance.shape[0]
        flt_focal = 886.81  # focal angle:60, 1024 / (2 * np.tan(np.radians(60) / 2))
        image_plane_x = np.linspace((-0.5 * int_width) + 0.5, (0.5 * int_width) - 0.5, int_width).reshape(1, int_width).repeat(int_height, 0).astype(np.float32)[:, :, None]
        image_plane_y = np.linspace((-0.5 * int_height) + 0.5, (0.5 * int_height) - 0.5, int_height).reshape(int_height, 1).repeat(int_width, 1).astype(np.float32)[:, :, None]
        image_plane_z = np.full([int_height, int_width, 1], flt_focal, np.float32)
        image_plane = np.concatenate([image_plane_x, image_plane_y, image_plane_z], 2)
        depth = cam_distance / np.linalg.norm(image_plane, 2, 2, keepdims=True) * flt_focal

        # # Convert depth to point cloud
        # point_cloud = depth_map_to_point_cloud(depth, fov=60)
        # # Save to XYZ file
        # np.savetxt('test_hypersim/point_cloud_0.xyz', point_cloud.reshape(-1, 3), delimiter=' ', fmt='%.6f')
        # point_cloud_rgb = depth_map_to_point_cloud_with_rgb(depth, image**(1/2.2), fov=60)
        # # Save to XYZRGB file
        # np.savetxt('test_hypersim/point_cloud_w_rgb_0.txt', point_cloud_rgb.reshape(-1, 6), delimiter=' ', fmt='%.6f')
        # point_distance = compute_distance_bgpc_objpc(point_cloud.reshape(-1, 3), point_cloud.reshape(-1, 3))

        # Load normal, normal valid range [-1,1]
        normal = load_image(normal_path, isGamma=False)

        # Load diffuse, diffuse valid range [0,1]
        diffuse = load_hdf5(diffuse_path)

        # shading = compute_shading(image, diffuse)

        sample = {
            'pixel_values': image,
            # 'cam_distance': cam_distance,
            'depth': depth,
            'normal': normal,
            'diffuse': diffuse,
            # 'shading': shading,
        }

        # Apply transforms
        if self.transforms:
            sample = self.transforms(sample)

        # Apply color transforms
        if self.color_transforms:
            image = self.color_transforms(sample['pixel_values'])
        else:
            image = sample['pixel_values'] ** (1.0 / 2.2)
        sample['pixel_values'] = torch.clamp(image, 0, 1)

        # Compute valid mask
        depth = sample['depth']
        depth_valid_mask = (depth > 0).float()
        sample['depth_valid_mask'] = depth_valid_mask

        # sample['valid_mask'] = depth_valid_mask * diffuse_nozero_mask

        # Convert to controlnet input
        if self.to_controlnet_input:
            tci_sample = self.to_controlnet_input({'pixel_values': sample['pixel_values'].clip(0, 1)})
            sample.update(tci_sample)
        return sample


def load_image(image_name, isGamma=False):
    image = cv2.imread(image_name, -1)
    image = np.asarray(image, dtype=np.float32)

    image = image / 255.0
    if isGamma:
        image = image**2.2
    if len(image.shape) == 2:
        image = image[:, :, np.newaxis]
    if len(image.shape) == 3:
        if image.shape[-1] == 4:
            image = image[:, :, :3]
        image = image[:, :, ::-1]

    return np.ascontiguousarray(image)


def load_hdf5(path):
    with h5py.File(path, 'r') as f:
        data = f['dataset'][:]
    return data


if __name__ == "__main__":
    from torchvision.transforms import v2
    from torchvision.transforms import functional as F
    from torchvision.tv_tensors import Image, Mask
    from PIL import Image
    from data_augmentation import RandomGammaCorrection
    from torch.utils.data import DataLoader
    from tqdm import tqdm

    train_transforms = v2.Compose([
        v2.ToImage(),
        v2.Resize(size=[512, ], interpolation=v2.InterpolationMode.NEAREST_EXACT),
        v2.RandomCrop([512, 512]),
        v2.RandomHorizontalFlip(p=0.5)
    ])
    color_transforms = v2.Compose([
        RandomGammaCorrection(gamma_range=(1.8, 2.4)),
        # v2.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.1, hue=0.1)
    ])
    dataset = HypersimDataset("../datasets/hypersim", 'train', train_transforms, color_transforms)
    # dataloader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=4)

    def tensor_to_np(img, initial_range=(0, 1)):
        # scale to [0, 1]
        img = img - initial_range[0]
        img = img / (initial_range[1] - initial_range[0])
        return np.clip(img.permute(1, 2, 0).cpu().numpy(), 0, 1)

    def np_to_pil(img):
        img = (img * 255.0).astype("uint8")
        if img.shape[-1] == 1:
            # special case for grayscale (single channel) images
            pil_img = Image.fromarray(img.squeeze(), mode="L")
        else:
            pil_img = Image.fromarray(img, mode="RGB")
        return pil_img

    def tensor_to_pil(img, initial_range=(0, 1)):
        img = tensor_to_np(img, initial_range)
        img = np_to_pil(img)
        return img

    for i, batch in enumerate(tqdm(dataset)):
        for k, v in batch.items():
            if k == 'depth' or k == 'cam_distance':
                tensor_to_pil(v, (v[v > 0].min(), v.max())).save(f'test_hypersim/{k}_{i}.png')
            else:
                tensor_to_pil(v).save(f'test_hypersim/{k}_{i}.png')

            # if 'path' not in k:
            #     assert not torch.isnan(v).any(), f"{k} is nan!!!!!!!" + batch['depth_path']
            # break
        break
