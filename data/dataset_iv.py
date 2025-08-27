from torch.utils.data import Dataset
import torch
import os
import glob
import struct
import numpy as np
import cv2
import ezexr

EPS = 1e-6


def compute_shading(img, diffuse):
    shading = np.ones_like(diffuse) * EPS
    # shading = torch.ones_like(diffuse) * -1
    diffuse_nozero_mask = diffuse > EPS
    shading[diffuse_nozero_mask] = img[diffuse_nozero_mask] / diffuse[diffuse_nozero_mask]

    shading = shading.clip(EPS, 1e3)

    # shading = img / diffuse # It will lead to super large values
    return shading


class InteriorVerseDataset(Dataset):
    def __init__(self, root_dir, mode='train', transforms=None, color_transforms=None, to_controlnet_input=None):
        self.root_dir = root_dir
        self.mode = mode
        self.transforms = transforms
        self.color_transforms = color_transforms
        self.to_controlnet_input = to_controlnet_input

        im_paths = glob.glob(os.path.join(self.root_dir, '*', '*_im.exr'))

        self.directory_files = {
            'image': im_paths,
            'depth': [p.replace('_im.exr', '_depth.exr') for p in im_paths],
            'normal': [p.replace('_im.exr', '_normal.exr') for p in im_paths],
            'diffuse': [p.replace('_im.exr', '_albedo.exr') for p in im_paths],
            'material': [p.replace('_im.exr', '_material.exr') for p in im_paths]
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

        # print(f"The length of the dataset is {len(self.directory_files['seg'])}")
        for k, v in self.directory_files.items():
            self.directory_files[k] = [v[i] for i in range(len(v)) if i not in lack_list]
            for p in self.directory_files[k]:
                assert os.path.isfile(p), f'{p} does not exist'

        # print(f"The length of the dataset is {len(self.directory_files['seg'])}")

    def __len__(self):
        return len(self.directory_files['image'])

    def __getitem__(self, idx):
        image_path = self.directory_files['image'][idx]
        depth_path = self.directory_files['depth'][idx]
        normal_path = self.directory_files['normal'][idx]
        diffuse_path = self.directory_files['diffuse'][idx]
        material_path = self.directory_files['material'][idx]

        name = os.path.dirname(image_path) + '_' + os.path.basename(image_path).split('.')[0][:3]

        # Load depth
        depth = load_exr(depth_path)[:, :, :1]
        # Find inf values as a mask
        depth_invalid_mask = np.isinf(depth)
        # depth_invalid_mask_3 = np.repeat(depth_invalid_mask, 3, axis=2)
        depth_valid_mask = ~depth_invalid_mask
        depth_valid_mask = depth_valid_mask.astype(np.float32)
        # replace them with 0
        depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
        # Convert depth from millimeters to meters
        depth = depth[:, :, 0:1] / 1000.0

        # Load rgb image
        image = load_exr(image_path)
        # image[depth_invalid_mask_3] = 0.0
        image = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)

        # Load normal
        normal = load_exr(normal_path)
        # normal[depth_invalid_mask_3] = 0.0
        normal = np.nan_to_num(normal, nan=0.0, posinf=0.0, neginf=0.0)
        # Normalize
        normal = normal / (np.linalg.norm(normal, axis=2, keepdims=True) + EPS)
        # Convert normal from [-1, 1] to [0, 1]
        normal = (normal + 1.0) / 2.0

        # Load diffuse
        diffuse = load_exr(diffuse_path)
        diffuse = np.nan_to_num(diffuse, nan=0.0, posinf=0.0, neginf=0.0)

        # Load roughness and metallic
        material = load_exr(material_path)
        material = np.nan_to_num(material, nan=0.0, posinf=0.0, neginf=0.0)
        roughness = material[:, :, 0:1]
        metallic = material[:, :, 1:2]

        # shading = compute_shading(image, diffuse)

        sample = {
            'pixel_values': image,
            'depth_valid_mask': depth_valid_mask,
            'depth': depth,
            'normal': normal,
            'diffuse': diffuse,
            'roughness': roughness,
            'metallic': metallic,
            'name': name,
            # 'shading': shading
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

        if self.to_controlnet_input:
            tci_sample = self.to_controlnet_input({'pixel_values': sample['pixel_values']})
            sample.update(tci_sample)

        return sample


def load_exr(path):
    img = ezexr.imread(path, rgb=True)
    img = np.asarray(img, dtype=np.float32)
    return img


if __name__ == "__main__":
    from torchvision.transforms import v2
    from torchvision.transforms import functional as F
    from PIL import Image
    from data_augmentation import RandomGammaCorrection
    from tqdm import tqdm

    train_transforms = v2.Compose([
        v2.ToImage(),
        v2.Resize(size=[512, ], antialias=True),
        v2.RandomCrop([512, 512]),
        # v2.RandomHorizontalFlip(p=0.5)
    ])

    color_transforms = v2.Compose([
        v2.ColorJitter(brightness=0.2),
        RandomGammaCorrection(gamma_range=(2.0, 2.25)),
    ])
    dataset = InteriorVerseDataset("../datasets/interior_verse", 'train', train_transforms, color_transforms)

    def tensor_to_numpy(img, initial_range=(0, 1)):
        # scale to [0, 1]
        img = img - initial_range[0]
        img = img / (initial_range[1] - initial_range[0])
        return np.clip(img.permute(1, 2, 0).cpu().numpy(), 0, 1)

    def numpy_to_pil(img):
        img = (img * 255.0).astype("uint8")
        if img.shape[-1] == 1:
            # special case for grayscale (single channel) images
            pil_img = Image.fromarray(img.squeeze(), mode="L")
        else:
            pil_img = Image.fromarray(img, mode="RGB")
        return pil_img

    def tensor_to_pil(img, initial_range=(0, 1)):
        img = tensor_to_numpy(img, initial_range)
        img = numpy_to_pil(img)
        return img

    for i, batch in enumerate(tqdm(dataset)):
        for k, v in batch.items():
            if k == 'depth':
                tensor_to_pil(v, (v.min(), v.max())).save(f'test_iv/{k}_{i}.png')
            elif k == 'name':
                continue
            else:
                tensor_to_pil(v).save(f'test_iv/{k}_{i}.png')
        # break
