from torch.utils.data import Dataset
import torch
import os
import glob
from PIL import Image
import numpy as np
from hdrio import ezexr
import cv2


class LaboDataset(Dataset):
    def __init__(self, root_dir, transforms=None, to_controlnet_input=None):
        # Resize: If size is an int, the smaller edge of the image will be matched to this number maintaining the aspect ratio.
        self.transforms = transforms
        self.to_controlnet_input = to_controlnet_input
        # One exr file contains all the images
        self.directory_files = sorted(glob.glob(os.path.join(root_dir, '*', '*bundle*.exr')))

        lack_list = []
        # Check if file exists
        assert len(self.directory_files) > 0 and len(self.directory_files) > 0, f'Filelist is empty!'
        for i, p in enumerate(self.directory_files):
            if not os.path.isfile(p):
                print(f'{p} does not exist, deprecating...')
                if i not in lack_list:
                    lack_list.append(i)

        self.directory_files = [p for i, p in enumerate(self.directory_files) if i not in lack_list]

        for p in self.directory_files:
            assert os.path.isfile(p), f'{p} does not exist'

    def __len__(self):
        return len(self.directory_files)

    def __getitem__(self, idx):
        # Source provide the object, destination provide the background
        src_exr = ezexr.imread(self.directory_files[idx], rgb="hybrid")
        # Get name of the file
        name = os.path.basename(self.directory_files[idx]).split('.')[0][:-11]
        # object brdf
        src_diffuse = src_exr['albedo'][:, :, :3]
        src_roughness = src_exr['roughness'][:, :, :1]
        src_metallic = src_exr['metallic'][:, :, :1]
        # debug save diffuse
        # ezexr.imwrite(f'./tmp/diffuse_{idx}.exr', src_diffuse)
        # cv2.imwrite(f'./tmp/diffuse_{idx}.png', cv2.cvtColor((src_diffuse ** (1/2.2) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))

        src_depth = src_exr['depth'][:, :, :1]
        src_footprint_depth = src_exr['footprint_depth'][:, :, :1]
        src_mask = src_exr['mask']
        src_normal = src_exr['normals'][:, :, :3]
        src_normal = comp_normal_to_openrooms_normal(src_normal)

        src_obj = gamma_correction(src_exr['foreground'][:, :, :3])
        dst_bg = gamma_correction(src_exr['background'][:, :, :3])
        dst_comp = gamma_correction(src_exr['composite'][:, :, :3])

        sample = {
            'name': name,
            'depth': src_depth,
            'normal': src_normal,
            'diffuse': src_diffuse,
            'roughness': src_roughness,
            'metallic': src_metallic,
            'pixel_values': dst_bg,
            'mask': src_mask,
            'src_obj': src_obj,
            'comp': dst_comp,
            'footprint_depth': src_footprint_depth,
        }

        intrinsic_name = name
        # Rebuttal
        bg_depth_path = os.path.join(os.path.dirname(self.directory_files[idx]), intrinsic_name + '_bg_depth.exr')
        if os.path.exists(bg_depth_path):
            bg_depth = ezexr.imread(bg_depth_path)[:, :, :1]
            sample['bg_depth'] = bg_depth
        else:
            print(f'{bg_depth_path} does not exist')

        bg_normal_path = os.path.join(os.path.dirname(self.directory_files[idx]), intrinsic_name + '_bg_normal.png')
        if os.path.exists(bg_normal_path):
            bg_normal = load_image(bg_normal_path, isGamma=False)
            sample['bg_normal'] = bg_normal
        else:
            print(f'{bg_normal_path} does not exist')

        bg_diffuse_path = os.path.join(os.path.dirname(self.directory_files[idx]), intrinsic_name + '_bg_diffuse.png')
        if os.path.exists(bg_diffuse_path):
            bg_diffuse = load_image(bg_diffuse_path, isGamma=False)
            sample['bg_diffuse'] = bg_diffuse
        else:
            print(f'{bg_diffuse_path} does not exist')

        bg_roughness_path = os.path.join(os.path.dirname(self.directory_files[idx]), intrinsic_name + '_bg_roughness.png')
        if os.path.exists(bg_roughness_path):
            bg_roughness = load_image(bg_roughness_path, isGamma=False)
            sample['bg_roughness'] = bg_roughness
        else:
            print(f'{bg_roughness_path} does not exist')

        bg_metallic_path = os.path.join(os.path.dirname(self.directory_files[idx]), intrinsic_name + '_bg_metallic.png')
        if os.path.exists(bg_metallic_path):
            bg_metallic = load_image(bg_metallic_path, isGamma=False)
            sample['bg_metallic'] = bg_metallic
        else:
            print(f'{bg_metallic_path} does not exist')

        # obj_diffuse_path = os.path.join(os.path.dirname(self.directory_files[idx]), intrinsic_name + '_comp_albedo.png')
        # if os.path.exists(obj_diffuse_path):
        #     obj_diffuse = load_image(obj_diffuse_path, isGamma=False)
        #     sample['diffuse'] = obj_diffuse

        # obj_roughness_path = os.path.join(os.path.dirname(self.directory_files[idx]), intrinsic_name + '_comp_roughness.png')
        # if os.path.exists(obj_roughness_path):
        #     obj_roughness = load_image(obj_roughness_path, isGamma=False)
        #     sample['roughness'] = obj_roughness
        # else:
        #     print(f'{obj_roughness_path} does not exist')

        # obj_metallic_path = os.path.join(os.path.dirname(self.directory_files[idx]), intrinsic_name + '_comp_metallic.png')
        # if os.path.exists(obj_metallic_path):
        #     obj_metallic = load_image(obj_metallic_path, isGamma=False)
        #     sample['metallic'] = obj_metallic
        # else:
        #     print(f'{obj_metallic_path} does not exist')

        if self.transforms:
            sample = self.transforms(sample)

        if self.to_controlnet_input:
            sample = self.to_controlnet_input(sample)

        return sample


def gamma_correction(img, gamma=2.2):
    return (img.clip(0, 1) ** (1 / gamma)).clip(0, 1)


def comp_normal_to_openrooms_normal(normal):
    # Input normal map should be in [-1, 1] range
    if normal.min() >= 0:
        normal = normal * 2 - 1
    normal[:, :, 2] = -normal[:, :, 2]
    # Transform it back to [0, 1] range
    normal = normal * 0.5 + 0.5
    return normal


def load_image(image_name, isGamma=False):
    image = cv2.imread(image_name, -1)
    image = np.asarray(image, dtype=np.float32)

    image = image / 255.0
    if isGamma:
        image = image**2.2
    if len(image.shape) == 2:
        image = image[:, :, np.newaxis]
    if len(image.shape) == 3:
        image = image[:, :, ::-1]

    return np.ascontiguousarray(image)


if __name__ == '__main__':
    dataloader = LaboDataset('../datasets/labo/GT_emission_envmap')
    save_dir = '../datasets/labo_png'
    from torch.utils.data import DataLoader
    import cv2
    from PIL import Image
    dataloader = DataLoader(dataloader, batch_size=1, shuffle=False, num_workers=4)

    def tensor_to_numpy(img, initial_range=(0, 1)):
        # scale to [0, 1]
        img = img - initial_range[0]
        img = img / (initial_range[1] - initial_range[0])
        if img.dim() == 4:
            img = img.squeeze(0)
        if img.shape[0] <= 3:
            img = img.permute(1, 2, 0)
        return np.clip(img.cpu().numpy(), 0, 1)

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

    import time
    from tqdm import tqdm
    t1 = time.time()
    for i, sample in enumerate(tqdm(dataloader)):
        pass
