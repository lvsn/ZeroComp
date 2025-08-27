from torch.utils.data import Dataset
import torch
import os
import glob
import struct
import numpy as np
import cv2


class OpenroomsAllDataset(Dataset):
    def __init__(self, root_dir, mode='train', transforms=None, color_transforms=None, to_controlnet_input=None):
        self.root_dir = root_dir
        self.mode = mode
        self.transforms = transforms
        self.color_transforms = color_transforms
        self.to_controlnet_input = to_controlnet_input

        # scene_names = [l.strip('\n') for l in (open(os.path.join('inputs', f'{mode}.txt')).readlines())]
        # TODO: add support for splits train/val/test
        # image_paths = glob.glob(os.path.join(self.root_dir, mode, 'Image', '**', 'im_*.hdr'), recursive=True)
        image_paths = glob.glob(os.path.join(self.root_dir, 'Image', '**', 'im_*.hdr'), recursive=True)
        # image_paths = glob.glob(os.path.join(self.root_dir, 'Denoised', '**', 'im_*.hdr'), recursive=True)
        image_scene_names = [os.path.basename(os.path.dirname(p)) for p in image_paths]
        image_file_numbers = [os.path.basename(p).split('_')[-1].split('.')[0] for p in image_paths]

        # Some folder names (DiffLight, DiffMat) are renamed to the canonical name, because the dataset doesn't duplicate the data points with duplicated intrinsics
        self.directory_files = {
            'image': image_paths,
            'seg': [os.path.join(root_dir, 'Mask', image_scene_names[i].replace('mainDiffMat', 'main'), f'immask_{image_file_numbers[i]}.png') for i in range(len(image_paths))],
            'depth': [os.path.join(root_dir, 'Geometry', image_scene_names[i].replace('mainDiffLight', 'main').replace('mainDiffMat', 'main'), f'imdepth_{image_file_numbers[i]}.dat') for i in range(len(image_paths))],
            'normal': [os.path.join(root_dir, 'Geometry', image_scene_names[i].replace('mainDiffLight', 'main'), f'imnormal_{image_file_numbers[i]}.png') for i in range(len(image_paths))],
            'diffuse': [os.path.join(root_dir, 'Material', image_scene_names[i].replace('mainDiffLight', 'main'), f'imbaseColor_{image_file_numbers[i]}.png') for i in range(len(image_paths))],
            'roughness': [os.path.join(root_dir, 'Material', image_scene_names[i].replace('mainDiffLight', 'main'), f'imroughness_{image_file_numbers[i]}.png') for i in range(len(image_paths))],
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
        scene_name = os.path.basename(os.path.dirname(self.directory_files['seg'][idx]))
        file_name = os.path.basename(self.directory_files['seg'][idx]).split('_')[-1].split('.')[0]
        name = f"{scene_name}_{file_name}"
        seg_path = self.directory_files['seg'][idx]
        image_path = self.directory_files['image'][idx]
        depth_path = self.directory_files['depth'][idx]
        normal_path = self.directory_files['normal'][idx]
        diffuse_path = self.directory_files['diffuse'][idx]
        roughness_path = self.directory_files['roughness'][idx]

        seg_image = load_image(seg_path)[:, :, 0:1]
        seg = seg_image

        # Load image, [:,:,::-1] converts BGR to RGB
        hdr_image = load_hdr(image_path, b_bgr2rgb=True)
        scale = scale_hdr(hdr_image, seg_image, self.mode)

        # image = self.scale_hdr_reinhard(hdr_image)
        image = hdr_image * scale
        image = np.clip(image, 0, 1.0)

        # Load depth, depth valid range [1,10]
        with open(depth_path, 'rb') as fIn:
            # Read the height and width of depth
            hBuffer = fIn.read(4)
            height = struct.unpack('i', hBuffer)[0]
            wBuffer = fIn.read(4)
            width = struct.unpack('i', wBuffer)[0]
            # Read depth
            dBuffer = fIn.read(4 * width * height)
            depth = np.array(
                struct.unpack('f' * height * width, dBuffer),
                dtype=np.float32)
            depth = depth.reshape(height, width)
            depth = np.expand_dims(depth, axis=2)
        # depth = (depth - depth.min()) / (depth.max() - depth.min())

        # depth = sdi_utils.numpy_to_pil(depth))
        # depth.save(f'test_openrooms/depth_{i}.png')

        normal = load_image(normal_path, isGamma=False)
        normal = 2.0 * normal - 1.0
        normal = normal / np.sqrt(np.maximum(np.sum(normal*normal, axis=-1, keepdims=True), 1e-5))
        normal = (normal + 1.0) / 2.0

        diffuse = load_image(diffuse_path, isGamma=True)

        # roughness = load_image(roughness_path, isGamma=False)

        sample = {
            'name': name,
            'pixel_values': image,
            'depth_valid_mask': seg,
            'depth': depth,
            'normal': normal,
            'diffuse': diffuse,
            # 'roughness': roughness
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
        seg = sample['depth_valid_mask']
        depth_valid_mask = (seg > 0.9).float()
        sample['depth_valid_mask'] = depth_valid_mask

        if self.to_controlnet_input:
            tci_sample = self.to_controlnet_input({'pixel_values': sample['pixel_values'].clip(0, 1)})
            sample.update(tci_sample)

        return sample


def get_number(path):
    return path.split('_')[-1].split('.')[0]


def load_hdr(image_name, b_bgr2rgb=True):
    # assert os.path.isfile(image_name)
    image = cv2.imread(image_name, -1)
    if b_bgr2rgb:
        image = image[:, :, ::-1]
    # image = np.transpose(image, [2, 0, 1])
    return np.ascontiguousarray(image)


def scale_hdr(hdr, seg, mode='test'):
    intensity_arr = (hdr*seg).flatten()
    intensity_arr.sort()
    if mode == 'train':
        scale = (0.95 - 0.1 * np.random.random()) / np.clip(intensity_arr[int(0.95 * len(intensity_arr))], 0.1, None)
    else:
        scale = (0.95 - 0.05) / np.clip(intensity_arr[int(0.95 * len(intensity_arr))], 0.1, None)

    return scale


def load_image(image_name, isGamma=False):
    image = cv2.imread(image_name, -1)
    image = np.asarray(image, dtype=np.float32)

    image = image/255.0
    if isGamma:
        image = image**2.2
    if len(image.shape) == 2:
        image = image[:, :, np.newaxis]
    if len(image.shape) == 3:
        image = image[:, :, ::-1]

    return np.ascontiguousarray(image)


def to_tensor(img):
    img = torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1)))
    return img


if __name__ == "__main__":
    from torchvision.transforms import v2
    from torchvision.transforms import functional as F
    from PIL import Image
    from data_augmentation import RandomGammaCorrection
    from tqdm import tqdm
    from torch.utils.data import DataLoader

    train_transforms = v2.Compose([
        v2.ToImage(),
        v2.Resize(size=[512, ], interpolation=v2.InterpolationMode.BILINEAR, antialias=True),
        v2.RandomCrop([512, 512]),
        v2.RandomHorizontalFlip(p=0.5)
    ])
    color_transforms = v2.Compose([
        v2.ColorJitter(brightness=0.2),
        RandomGammaCorrection(gamma_range=(2.0, 2.25)),
    ])

    def collate_fn(examples):
        stacked_examples = {}
        for k, _ in examples[0].items():
            if k == 'input_ids':
                stacked_examples[k] = torch.stack([example[k] for example in examples])
            elif k == 'caption' or k == 'name':
                stacked_examples[k] = [example[k] for example in examples]
            else:
                batched = torch.stack([example[k] for example in examples])
                stacked_examples[k] = batched.to(memory_format=torch.contiguous_format)

        return stacked_examples
    dataset = OpenroomsAllDataset("../datasets/openrooms_all_filtered", 'train', train_transforms, color_transforms)

    train_dataloader = DataLoader(
        dataset,
        shuffle=True,
        batch_size=4,
        num_workers=8,
        collate_fn=collate_fn
    )

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

    for i, batch in enumerate(tqdm(train_dataloader)):
        for k, v in batch.items():
            mean_values = batch["pixel_values"].mean(dim=(1, 2, 3))
            name = batch["name"]
            black_images = mean_values < 0.01
            if black_images.any():
                print(f"Black image detected in the batch! {name}")
                continue
            print(k, v.shape)
        
        break