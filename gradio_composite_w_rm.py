from datetime import datetime
import os

import torch
import torch.utils.checkpoint
from packaging import version
from torchvision.transforms import v2
from torchvision.transforms.v2 import functional as tf
from torchvision.ops import masks_to_boxes
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import numpy as np

from diffusers import (
    AutoencoderKL,
    ControlNetModel,
    StableDiffusionControlNetPipeline,
    DDIMScheduler,
    UNet2DConditionModel,
)
from diffusers.utils.import_utils import is_xformers_available
from diffusers.utils.torch_utils import randn_tensor

from controlnet_input_handle import ToControlNetInput, ToPredictors, compute_shading, match_depth_from_footprint
# from models.controlnet import ControlNetModel
# from pipeline_controlnet import StableDiffusionControlNetPipeline

import sdi_utils
from sdi_utils import import_model_class_from_model_name_or_path
import gradio as gr
import ezexr

EPS = 1e-6

# Arguments
pretrained_model_name_or_path = ".cache/huggingface/hub/models--stabilityai--stable-diffusion-2-1/snapshots/5cae40e6a2745ae2b01ad92ae5043f95f23644d6"
controlnet_model_name_or_path = "checkpoints/interior_verse_2days"
revision = None
bs = 1
resolution = 512
vis_resolution = 256
device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cpu":
    raise ValueError("This running requires GPU acceleration.")
weight_dtype = torch.float16
# weight_dtype = torch.float32
gradio_dir = './gradio_cache'
gradio_save_dir = './results_gradio'

# Global variables
g_obj_batch = None
g_dst_batch = None
g_comp_batch = None


# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    pretrained_model_name_or_path,
    subfolder="tokenizer",
    revision=revision,
    use_fast=False,
)

# import correct text encoder class
text_encoder_cls = import_model_class_from_model_name_or_path(pretrained_model_name_or_path, revision)

# Load scheduler and models
text_encoder = text_encoder_cls.from_pretrained(
    pretrained_model_name_or_path, subfolder="text_encoder", revision=revision
)
vae = AutoencoderKL.from_pretrained(pretrained_model_name_or_path, subfolder="vae", revision=revision)
unet = UNet2DConditionModel.from_pretrained(
    pretrained_model_name_or_path, subfolder="unet", revision=revision
)

controlnet = ControlNetModel.from_pretrained(controlnet_model_name_or_path, subfolder="controlnet")

vae.requires_grad_(False)
unet.requires_grad_(False)
text_encoder.requires_grad_(False)
controlnet.requires_grad_(False)
controlnet.eval()

if is_xformers_available():
    import xformers

    xformers_version = version.parse(xformers.__version__)
    unet.enable_xformers_memory_efficient_attention()
    controlnet.enable_xformers_memory_efficient_attention()
else:
    raise ValueError("xformers is not available. Make sure it is installed correctly")

to_controlnet_input = ToControlNetInput(
    device=device,
    feed_empty_prompt=True,
    tokenizer=tokenizer,
    for_sdxl=False
)

aug_cutout = None

to_predictors = ToPredictors(device,
                             True,
                             aug_cutout)

val_transforms = v2.Compose([
    v2.ToImage(),
    v2.Resize(size=[resolution, ], antialias=True),
    v2.CenterCrop([resolution, resolution])
])

# Move vae, unet and text_encoder to device
vae.to(device, dtype=weight_dtype)
unet.to(device, dtype=weight_dtype)
text_encoder.to(device, dtype=weight_dtype)
controlnet.to(device, dtype=weight_dtype)

pipeline = StableDiffusionControlNetPipeline.from_pretrained(
    pretrained_model_name_or_path,
    vae=vae,
    text_encoder=text_encoder,
    tokenizer=tokenizer,
    unet=unet,
    controlnet=controlnet,
    safety_checker=None,
    revision=revision
)
pipeline.scheduler = DDIMScheduler.from_config(
    pipeline.scheduler.config,
    timestep_spacing='trailing',
    rescale_betas_zero_snr=True
)
pipeline = pipeline.to(device, dtype=weight_dtype)
pipeline.set_progress_bar_config(disable=True)

pipeline.enable_xformers_memory_efficient_attention()

# os.makedirs(gradio_dir, exist_ok=True)


def load_image_from_gradio(image):
    image = np.ascontiguousarray(image, dtype=np.float32) / 255.0
    image = val_transforms(image)
    image = image.unsqueeze(0).to(device)
    if len(image.shape) == 3:
        image = image.unsqueeze(0)
    image = image.clip(0.0, 1.0)
    return image


@torch.inference_mode()
def predict_bg_intrinsics(background_rgb,
                          background_dp, background_nm,
                          background_df,
                          background_rg, background_mt):
    background_rgb = background_rgb[:, :, 0:3]
    background_rgb = load_image_from_gradio(background_rgb)
    temp_bg_batch = to_predictors({'pixel_values': background_rgb})

    controlnet_input = temp_bg_batch['controlnet_inputs']
    bg_depth_hdr, bg_normal, bg_diffuse, bg_shading_hdr = controlnet_input['depth'], controlnet_input['normal'], controlnet_input['diffuse'], controlnet_input['shading']

    if background_dp is not None:
        background_dp = background_dp[:, :, 0:1]
        background_dp = load_image_from_gradio(background_dp)
        bg_depth_hdr = background_dp

    if background_nm is not None:
        background_nm = background_nm[:, :, 0:3]
        background_nm = load_image_from_gradio(background_nm)
        bg_normal = background_nm

    if background_df is not None:
        background_df = background_df[:, :, 0:3]
        background_df = load_image_from_gradio(background_df)
        bg_diffuse = background_df
        # Recompute shading
        bg_shading_hdr = compute_shading(background_rgb, bg_diffuse)

    if background_rg is not None:
        background_rg = background_rg[:, :, 0:1]
        background_rg = load_image_from_gradio(background_rg)
    else:
        background_rg = torch.ones_like(bg_depth_hdr) * 1.0

    if background_mt is not None:
        background_mt = background_mt[:, :, 0:1]
        background_mt = load_image_from_gradio(background_mt)
    else:
        background_mt = torch.zeros_like(bg_depth_hdr)

    temp_bg_batch = {
        'depth': bg_depth_hdr,
        'normal': bg_normal,
        'diffuse': bg_diffuse,
        'shading': bg_shading_hdr,
        'pixel_values': background_rgb,
        'caption': temp_bg_batch['caption'],
        'roughness': background_rg,
        'metallic': background_mt
    }

    # Save to global variable
    global g_dst_batch
    g_dst_batch = temp_bg_batch

    bg_rgb = sdi_utils.tensor_to_pil(background_rgb)
    bg_depth = sdi_utils.tensor_to_pil(bg_depth_hdr, initial_range=(bg_depth_hdr.min(), bg_depth_hdr.max()))
    bg_normal = sdi_utils.tensor_to_pil(bg_normal)
    bg_diffuse = sdi_utils.tensor_to_pil(bg_diffuse)
    bg_shading = sdi_utils.tensor_to_pil(bg_shading_hdr)
    bg_roughness = sdi_utils.tensor_to_pil(background_rg)
    bg_metallic = sdi_utils.tensor_to_pil(background_mt)

    return bg_rgb, bg_depth, bg_normal, bg_diffuse, bg_shading, bg_roughness, bg_metallic


@torch.inference_mode()
def process_obj_intrinsics(obj_file,
                           obj_img, mask_img,
                           obj_dp, obj_nm,
                           obj_df,
                           obj_rg, obj_mt,
                           object_intrinsic_mode, use_rgb_as_diffuse):
    if obj_file is not None:
        path = obj_file.name
        exr = ezexr.imread(path, rgb="hybrid")
        if object_intrinsic_mode == 'Load From Obj/Exr':
            # object brdf
            diffuse = exr['albedo'][:, :, :3]
            depth = exr['depth'][:, :, :1]
            normal = exr['normals'][:, :, :3]
            normal = sdi_utils.comp_normal_to_openrooms_normal(normal)
            mask = exr['mask']

            obj_img = exr['foreground'][:, :, :3]

            if 'roughness' in exr:
                roughness = exr['roughness'][:, :, :1]
            else:
                roughness = torch.ones_like(depth) * 1.0

            if 'metallic' in exr:
                metallic = exr['metallic'][:, :, :1]
            else:
                metallic = torch.zeros_like(depth)

            batch = {
                'diffuse': diffuse,
                'depth': depth,
                'normal': normal,
                'mask': mask,
                'src_obj': obj_img,
                'roughness': roughness,
                'metallic': metallic
            }

            for k, v in batch.items():
                batch[k] = val_transforms(v).unsqueeze(0).to(device)

        else:
            raise NotImplementedError("Not implemented yet.")

    elif obj_img is not None:
        obj_img = obj_img[:, :, 0:3]
        obj_img = load_image_from_gradio(obj_img)
        mask_img = mask_img[:, :, 0:1]
        mask = load_image_from_gradio(mask_img)

        batch = to_predictors({'pixel_values': obj_img})

        depth = batch['controlnet_inputs']['depth']
        normal = batch['controlnet_inputs']['normal']
        diffuse = batch['controlnet_inputs']['diffuse']
        batch = {
            'diffuse': diffuse,
            'depth': depth,
            'normal': normal,
            'mask': mask,
            'src_obj': obj_img
        }

    else:
        raise ValueError("No object image provided.")

    if use_rgb_as_diffuse:
        batch['diffuse'] = batch['src_obj']

    if obj_dp is not None:
        obj_dp = obj_dp[:, :, 0:1]
        obj_dp = load_image_from_gradio(obj_dp)
        batch['depth'] = obj_dp

    if obj_nm is not None:
        obj_nm = obj_nm[:, :, 0:3]
        obj_nm = load_image_from_gradio(obj_nm)
        batch['normal'] = obj_nm

    if obj_df is not None:
        obj_df = obj_df[:, :, 0:3]
        obj_df = load_image_from_gradio(obj_df)
        batch['diffuse'] = obj_df

    if obj_rg is not None:
        obj_rg = obj_rg[:, :, 0:1]
        obj_rg = load_image_from_gradio(obj_rg)
        batch['roughness'] = obj_rg

    if obj_mt is not None:
        obj_mt = obj_mt[:, :, 0:1]
        obj_mt = load_image_from_gradio(obj_mt)
        batch['metallic'] = obj_mt

    if 'roughness' not in batch:
        batch['roughness'] = torch.ones_like(batch['depth']) * 1.0
    if 'metallic' not in batch:
        batch['metallic'] = torch.zeros_like(batch['depth'])

    obj_vis = sdi_utils.tensor_to_pil(batch['src_obj'])
    depth_vis = sdi_utils.tensor_to_pil(batch['depth'], initial_range=(batch['depth'].min(), batch['depth'].max()))
    normal_vis = sdi_utils.tensor_to_pil(batch['normal'])
    diffuse_vis = sdi_utils.tensor_to_pil(batch['diffuse'])
    roughness_vis = sdi_utils.tensor_to_pil(batch['roughness'])
    metallic_vis = sdi_utils.tensor_to_pil(batch['metallic'])
    mask_vis = sdi_utils.tensor_to_pil(batch['mask'])
    diffuse_edit_vis = sdi_utils.tensor_to_pil(batch['diffuse'])

    # Save to global variable
    global g_obj_batch
    g_obj_batch = batch

    return obj_vis, depth_vis, normal_vis, diffuse_vis, roughness_vis, metallic_vis, mask_vis, diffuse_edit_vis


@torch.inference_mode()
def process_edit_intrinsics(obj_diffuse_edited):
    obj_diffuse_edited = load_image_from_gradio(obj_diffuse_edited['composite'][:, :, 0:3])
    obj_diffuse_edited = obj_diffuse_edited.to(device)

    # Modify the object batch
    global g_obj_batch
    g_obj_batch['diffuse'] = obj_diffuse_edited


@torch.inference_mode()
def process_comp(obj_relative_scale, obj_relative_vertical_position, obj_relative_horizontal_position,
                 obj_depth_min_value, obj_depth_scale,
                 shading_maskout_mode, shading_maskout_dilation, shading_maskout_range,
                 occlusion,
                 obj_diffuse_offset, obj_roughness_offset, obj_metallic_offset,):
    global g_obj_batch
    obj_depth = g_obj_batch['depth']
    obj_normal = g_obj_batch['normal']
    obj_diffuse = g_obj_batch['diffuse']
    obj_roughness = g_obj_batch['roughness']
    obj_metallic = g_obj_batch['metallic']
    obj_mask = g_obj_batch['mask']
    obj_src_obj = g_obj_batch['src_obj']
    obj_batch = {
        'depth': obj_depth,
        'normal': obj_normal,
        'diffuse': obj_diffuse,
        'mask': obj_mask,
        'roughness': obj_roughness,
        'metallic': obj_metallic,
        'src_obj': obj_src_obj
    }

    # Apply obj intrinsic offset
    obj_batch['diffuse'] = torch.clamp(obj_batch['diffuse'] + obj_diffuse_offset, 0, 1)
    obj_batch['roughness'] = torch.clamp(obj_batch['roughness'] + obj_roughness_offset, 0, 1)
    obj_batch['metallic'] = torch.clamp(obj_batch['metallic'] + obj_metallic_offset, 0, 1)

    # Apply obj relative position
    for k, v in obj_batch.items():
        obj_batch[k] = tf.affine(v, angle=0, translate=[obj_relative_horizontal_position, obj_relative_vertical_position], scale=obj_relative_scale, shear=0)

    # Apply obj depth scale
    obj_depth = obj_batch['depth']
    # Normalize the depth to [0, 1]
    obj_original_scale = obj_depth[obj_mask > 0.9].max() - obj_depth[obj_mask > 0.9].min()
    if obj_original_scale < EPS:
        obj_original_scale = 1
    obj_depth = (obj_depth - obj_depth[obj_mask > 0.9].min()) / (obj_original_scale + EPS)
    obj_depth = obj_depth * obj_depth_scale + obj_depth_min_value
    obj_batch['depth'] = obj_depth
    obj_mask = obj_batch['mask']

    global g_dst_batch
    # Get bg shading
    dst_bg = g_dst_batch['pixel_values'] * 0.5 + 0.5
    dst_depth = g_dst_batch['depth'].clone()
    dst_normal = g_dst_batch['normal']
    dst_diffuse = g_dst_batch['diffuse']
    dst_shading = g_dst_batch['shading']
    dst_roughness = g_dst_batch['roughness']
    dst_metallic = g_dst_batch['metallic']
    dst_mask = torch.ones_like(dst_depth)
    validation_prompt = g_dst_batch["caption"]

    # dst_depth = match_depth_from_footprint(dst_depth, obj_depth, obj_footprint_depth, adjust_background=True)

    dst_batch = {
        'depth': dst_depth,
        'normal': dst_normal,
        'diffuse': dst_diffuse,
        'shading': dst_shading,
        'mask': dst_mask,
        'roughness': dst_roughness,
        'metallic': dst_metallic,
    }

    # Apply occlusion
    if occlusion:
        occlusion_mask = torch.zeros_like(obj_mask)
        occlusion_mask[dst_depth > obj_depth] = 1
        obj_mask[obj_mask > 0.9] = occlusion_mask[obj_mask > 0.9]

        obj_batch['mask'] = obj_mask

    visualization = {}

    comp_batch = {}
    for k in dst_batch.keys():
        v = dst_batch[k].clone()
        tmp_mask = obj_mask.expand_as(v)
        if k == "shading":
            b = 0
            if shading_maskout_mode == 'None':
                v[tmp_mask > 0.9] = -1

            elif shading_maskout_mode == 'BBox':
                # Using dilated bounding box
                bbox = masks_to_boxes(obj_mask.squeeze(dim=1)).int()
                _, _, h, w = v.shape
                dilate_size = shading_maskout_dilation

                x1, y1, x2, y2 = bbox[0]
                x1 = x1 - dilate_size if x1 - dilate_size > 0 else 0
                y1 = y1 - dilate_size if y1 - dilate_size > 0 else 0
                x2 = x2 + dilate_size if x2 + dilate_size < w else w
                y2 = y2 + dilate_size if y2 + dilate_size < h else h
                v[0, :, y1:y2, x1:x2] = -1

            elif shading_maskout_mode == 'BBoxWithDepth':
                # Using dilated bounding box
                bbox = masks_to_boxes(obj_mask.squeeze(dim=1)).int()
                _, _, h, w = v.shape
                dilate_size = shading_maskout_dilation

                x1, y1, x2, y2 = bbox[0]
                x1 = x1 - dilate_size if x1 - dilate_size > 0 else 0
                y1 = y1 - dilate_size if y1 - dilate_size > 0 else 0
                x2 = x2 + dilate_size if x2 + dilate_size < w else w
                y2 = y2 + dilate_size if y2 + dilate_size < h else h
                v[0, :, y1:y2, x1:x2] = -1

                # If higher than a threshold, use the whole source background shading
                avg_obj_depth = obj_depth[b, :, :, :][obj_mask[b, :, :, :] > 0.9].mean()
                bg_depth = dst_depth[b, :, :, :]
                avg_obj_depth = avg_obj_depth.expand_as(bg_depth)
                out_of_depth_range_mask = torch.abs(bg_depth - avg_obj_depth) > shading_maskout_range

                out_of_depth_range_mask = torch.logical_and(out_of_depth_range_mask, ~(obj_mask[b, :, :, :].bool()))
                out_of_depth_range_mask = out_of_depth_range_mask.expand_as(dst_shading[b, :, :, :])
                v[b, out_of_depth_range_mask] = dst_shading[b, out_of_depth_range_mask]

            elif shading_maskout_mode == 'PointCloud':
                bg_depth = dst_depth[b, :, :, :]
                bg_point_cloud = sdi_utils.depth_map_to_point_cloud(bg_depth, fov=50).permute(1, 2, 0).reshape(-1, 3)
                obj_depth = obj_depth[b, :, :, :]
                obj_point_cloud = sdi_utils.depth_map_to_point_cloud(obj_depth, fov=50)
                obj_point_cloud = obj_point_cloud.permute(1, 2, 0)[(obj_mask[b, 0, :, :] > 0.9), :]
                dists = sdi_utils.compute_distance_bgpc_objpc(bg_point_cloud.cpu().numpy(), obj_point_cloud.cpu().numpy())
                dists = dists.reshape(bg_depth.shape[1], bg_depth.shape[2], 1)
                dists = torch.from_numpy(dists).to(device).permute(2, 0, 1)
                pc_crop_mask = None

                shading_maskout_pc_type = 'relative'
                if shading_maskout_pc_type == 'absolute':
                    pc_crop_mask = dists < shading_maskout_range
                elif shading_maskout_pc_type == 'relative':
                    object_height = obj_point_cloud[:, 1].max() - obj_point_cloud[:, 1].min()
                    pc_crop_mask = dists < object_height * shading_maskout_range
                else:
                    raise NotImplementedError

                _, _, h, w = v.shape
                shading_maskout_pc_above_cropping_type = 'argmin'
                if shading_maskout_pc_above_cropping_type == 'abovebbox':
                    bbox = masks_to_boxes(obj_mask.squeeze(dim=1)).int()
                    x1, y1, x2, y2 = bbox[b]
                    pc_crop_mask[:, :y1, :] = False
                elif shading_maskout_pc_above_cropping_type == 'argmin':
                    obj_mask_argmax = torch.argmax(obj_mask[b, :, :, :], dim=1, keepdim=True)
                    for j in range(w):
                        if obj_mask_argmax[0, 0, j] > 0:
                            pc_crop_mask[:, :obj_mask_argmax[0, 0, j], j] = False

                pc_crop_mask = pc_crop_mask.expand_as(dst_shading[b, :, :, :])
                obj_mask_b = obj_mask > 0.9
                crop_mask = torch.logical_or(pc_crop_mask, obj_mask_b[b, :, :, :])
                v[b, crop_mask] = -1

            elif shading_maskout_mode == 'Full':
                v = torch.ones_like(v) * -1
        else:
            v[tmp_mask > 0.9] = obj_batch[k][tmp_mask > 0.9]

        comp_batch[k] = v

    controlnet_inputs = []
    for k, v in comp_batch.items():
        if k == 'mask':
            v = torch.ones_like(v)
            shading = comp_batch['shading']
            v[shading[:, 0:1, :, :] == -1] = -1
            v = v.float()
            comp_batch[k] = v

        controlnet_inputs.append(v)

        if k == 'depth':
            visualization[k] = sdi_utils.tensor_to_pil_list(v, initial_range=(v.min(), v.max()))[0]
        else:
            visualization[k] = sdi_utils.tensor_to_pil_list(v)[0]

    conditioning = torch.cat(controlnet_inputs, dim=1)

    comp_batch = {
        'obj_mask': obj_mask,
        'conditioning': conditioning,
        'validation_prompt': validation_prompt,
        'comp_batch': comp_batch,
        'obj_batch': obj_batch,
    }

    # Save to global variable
    global g_comp_batch
    g_comp_batch = comp_batch

    return visualization['depth'], visualization['normal'], visualization['diffuse'], visualization['shading'], visualization['mask'], visualization['roughness'], visualization['metallic']


@torch.inference_mode()
def generate_image(seed, color_rebalance, post_compositing):
    global g_comp_batch
    comp_batch = g_comp_batch

    conditioning = comp_batch['conditioning']
    obj_mask = comp_batch['obj_mask']
    validation_prompt = comp_batch['validation_prompt']

    global g_dst_batch
    dst_batch = g_dst_batch
    dst_bg = dst_batch['pixel_values']
    dst_depth = dst_batch['depth']
    dst_normal = dst_batch['normal']
    dst_diffuse = dst_batch['diffuse']
    dst_shading = dst_batch['shading']

    dst_roughness = dst_batch['roughness']
    dst_metallic = dst_batch['metallic']

    obj_batch = comp_batch['obj_batch']

    generator = torch.Generator(device=device).manual_seed(seed)

    vis_batch = {}

    with torch.autocast("cuda"):
        noise_latents = randn_tensor([bs, pipeline.unet.config.in_channels, resolution // pipeline.vae_scale_factor, resolution // pipeline.vae_scale_factor],
                                     generator=generator, device=pipeline.device, dtype=pipeline.dtype)

        current_images = pipeline(
            validation_prompt, conditioning, num_inference_steps=20, generator=generator, latents=noise_latents, guidance_scale=0,
            output_type='pt'
        ).images
        current_images = torch.nan_to_num(current_images, nan=0, posinf=0, neginf=0)
        vis_batch['pred_comp'] = current_images[0]
        vis_batch['pred_bg'] = torch.zeros_like(current_images[0])
        vis_batch['post_comp'] = torch.zeros_like(current_images[0])

        if post_compositing:
            dst_mask = torch.ones_like(dst_depth)
            dst_conditioning = torch.cat([dst_depth, dst_normal, dst_diffuse, dst_shading, dst_mask], dim=1)

            dst_conditioning = torch.cat([dst_conditioning, dst_roughness, dst_metallic], dim=1)

            dst_images = pipeline(
                validation_prompt, dst_conditioning, num_inference_steps=20, generator=generator, latents=noise_latents, guidance_scale=0,
                output_type='pt'
            ).images

            dst_images = torch.nan_to_num(dst_images, nan=0, posinf=0, neginf=0)
            vis_batch['pred_bg'] = dst_images[0].clone()

            # Composite it back to original background image
            # Get the shadow area without the object
            comp_mask = comp_batch['comp_batch']['mask']

            # Calculate the visibility of the shadow area
            intensity_from_rgb = torch.tensor([0.299, 0.587, 0.114], device=device).view(1, 3, 1, 1)
            current_images_intensity = current_images * intensity_from_rgb
            dst_images_intensity = dst_images * intensity_from_rgb
            visibility = (current_images_intensity.sum(dim=1, keepdim=True) + EPS) / (dst_images_intensity.sum(dim=1, keepdim=True) + EPS)
            visibility = visibility.clamp(0, 1)

            visibility_mask = comp_mask.clamp(0, 1)
            visibility_mask = 1 - visibility_mask
            visibility_mask = v2.functional.gaussian_blur(visibility_mask, (15, 15), 1.5)
            visibility = visibility * visibility_mask + 1 * (1 - visibility_mask)
            vis_batch['visibility_mask'] = visibility_mask[0]
            vis_batch['visibility'] = visibility[0]
            # Dilate object mask
            obj_mask = obj_mask.float()
            vis_batch['obj_mask'] = obj_mask[0]
            # Do the color balance for the object
            # current_images_balanced = current_images * sdi_utils.color_rebalance(current_images, dst_bg)  # Buggy one
            if color_rebalance:
                current_images_balanced = current_images * sdi_utils.color_rebalance(dst_images, dst_bg)
            else:
                current_images_balanced = current_images
            compositing = dst_bg * visibility * (1 - obj_mask) + current_images_balanced * obj_mask
        else:
            compositing = current_images
        vis_batch['post_comp'] = compositing[0]

    # Save everything to the gradio savedir
    gradio_save_folder = os.path.join(gradio_save_dir, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    os.makedirs(gradio_save_folder, exist_ok=True)

    for k, v in vis_batch.items():
        v = sdi_utils.tensor_to_pil(v)
        v.save(os.path.join(gradio_save_folder, f"{k}.jpg"), quality=90)
        vis_batch[k] = v

    for k, v in obj_batch.items():
        if k == 'depth':
            v = sdi_utils.tensor_to_pil(v, initial_range=(v.min(), v.max()))
        else:
            v = sdi_utils.tensor_to_pil(v)
        v.save(os.path.join(gradio_save_folder, f"obj_{k}.jpg"), quality=90)

    for k, v in dst_batch.items():
        if not isinstance(v, torch.Tensor):
            continue
        if k == 'depth':
            v = sdi_utils.tensor_to_pil(v, initial_range=(v.min(), v.max()))
        else:
            v = sdi_utils.tensor_to_pil(v)
        v.save(os.path.join(gradio_save_folder, f"dst_{k}.jpg"), quality=90)

    for k, v in comp_batch['comp_batch'].items():
        if k == 'depth':
            v = sdi_utils.tensor_to_pil(v, initial_range=(v.min(), v.max()))
        else:
            v = sdi_utils.tensor_to_pil(v)
        v.save(os.path.join(gradio_save_folder, f"comp_{k}.jpg"), quality=90)

    return vis_batch['pred_comp'], vis_batch['pred_bg'], vis_batch['post_comp']


block = gr.Blocks().queue()
with block:
    with gr.Row():
        gr.Markdown("## Image Compositing")

    with gr.Row():
        with gr.Column():
            # background = gr.ImageEditor(sources=['upload'], type="numpy", height=vis_resolution, width=vis_resolution, crop_size=(512, 512))
            background_rgb = gr.Image(label="Bg Image", sources=['upload'], type="numpy", height=vis_resolution, width=vis_resolution)
            # background = gr.File(label="Background", type="numpy", file_count="single")

        with gr.Column():
            background_dp = gr.Image(label='Bg Depth', sources=['upload'], type="numpy", height=vis_resolution, width=vis_resolution)
            background_nm = gr.Image(label='Bg Normal', sources=['upload'], type="numpy", height=vis_resolution, width=vis_resolution)

        with gr.Column():
            background_df = gr.Image(label='Bg Diffuse', sources=['upload'], type="numpy", height=vis_resolution, width=vis_resolution)

        with gr.Column():
            background_rg = gr.Image(label='Bg Roughness', sources=['upload'], type="numpy", height=vis_resolution, width=vis_resolution)
            background_mt = gr.Image(label='Bg Metallic', sources=['upload'], type="numpy", height=vis_resolution, width=vis_resolution)

        with gr.Column():
            pred_bg_btn = gr.Button(value="Estimate intrinsics from background")
            # prompt = gr.Textbox(label="Prompt")

    with gr.Row():
        input_rgb_bg = gr.Image(label='Bg RGB', sources=['upload'], type="numpy", width=vis_resolution)
        input_depth_bg = gr.Image(label='Bg Depth', sources=['upload'], type="numpy", width=vis_resolution)
        input_normal_bg = gr.Image(label='Bg Normal', sources=['upload'], type="numpy", width=vis_resolution)
        input_shading_bg = gr.Image(label='Bg Shading', sources=['upload'], type="numpy", width=vis_resolution)
    with gr.Row():
        input_diffuse_bg = gr.Image(label='Bg Diffuse', sources=['upload'], type="numpy")
        input_roughness_bg = gr.Image(label='Bg Roughness', sources=['upload'], type="numpy", width=vis_resolution)
        input_metallic_bg = gr.Image(label='Bg Metallic', sources=['upload'], type="numpy", width=vis_resolution)

    with gr.Row():
        obj_file = gr.File(label="Obj Exr", file_count="single")

    with gr.Row():
        with gr.Column():
            obj_img = gr.Image(label='Obj Image', sources=['upload'], type="numpy", height=vis_resolution, width=vis_resolution)
            mask_img = gr.Image(label='Obj Mask', sources=['upload'], type="numpy", height=vis_resolution, width=vis_resolution)
        with gr.Column():
            obj_dp = gr.Image(label='Obj Depth', sources=['upload'], type="numpy", height=vis_resolution, width=vis_resolution)
            obj_nm = gr.Image(label='Obj Normal', sources=['upload'], type="numpy", height=vis_resolution, width=vis_resolution)
        with gr.Column():
            obj_df = gr.Image(label='Obj Diffuse', sources=['upload'], type="numpy", height=vis_resolution, width=vis_resolution)
        with gr.Column():
            obj_rg = gr.Image(label='Obj Roughness', sources=['upload'], type="numpy", height=vis_resolution, width=vis_resolution)
            obj_mt = gr.Image(label='Obj Metallic', sources=['upload'], type="numpy", height=vis_resolution, width=vis_resolution)
        with gr.Column():
            object_intrinsic_mode = gr.Radio(label='Object Intrinsic Mode', choices=['Load From Obj/Exr', 'Predict From RGB'], value='Predict From RGB')
            use_rgb_as_diffuse = gr.Checkbox(label='Use RGB as Diffuse', value=False)
            process_obj_btn = gr.Button(value="Process obj")

    with gr.Row():
        input_rgb_obj = gr.Image(label='Obj RGB', sources=['upload'], type="numpy", width=vis_resolution)
        input_depth_obj = gr.Image(label='Obj Depth', sources=['upload'], type="numpy", width=vis_resolution)
        input_normal_obj = gr.Image(label='Obj Normal', sources=['upload'], type="numpy", width=vis_resolution)
        input_mask_obj = gr.Image(label='Obj Mask', sources=['upload'], type="numpy", width=vis_resolution)

    with gr.Row():
        input_diffuse_obj = gr.Image(label='Obj Diffuse', sources=['upload'], type="numpy", width=vis_resolution)
        input_roughness_obj = gr.Image(label='Obj Roughness', sources=['upload'], type="numpy", width=vis_resolution)
        input_metallic_obj = gr.Image(label='Obj Metallic', sources=['upload'], type="numpy", width=vis_resolution)

    with gr.Row():
        input_diffuse_obj_edit = gr.ImageEditor(label='Obj Diffuse', sources=['upload'], type="numpy")

    with gr.Row():
        process_edit_btn = gr.Button(value="Edit Intrinsic")

    with gr.Row():
        with gr.Column():
            obj_relative_scale = gr.Slider(label="Obj Relative Scale", minimum=0.1, maximum=10.0, value=1.0, step=0.01)

            obj_relative_vertical_position = gr.Slider(label="Obj Relative Vertical Position", minimum=-512, maximum=512, value=0, step=1)
            obj_relative_horizontal_position = gr.Slider(label="Obj Relative Horizontal Position", minimum=-512, maximum=512, value=0, step=1)

        with gr.Column():
            obj_depth_min_value = gr.Slider(label="Obj Depth Min", minimum=0.0, maximum=10.0, value=1.0, step=0.01)
            obj_depth_scale = gr.Slider(label="Obj Depth Scale", minimum=0.01, maximum=10.0, value=1.0, step=0.01)

        with gr.Column():
            obj_diffuse_offset = gr.Slider(label="Obj Diffuse Offset", minimum=-1.0, maximum=1.0, value=0.0, step=0.01)
            obj_roughness_offset = gr.Slider(label="Obj Roughness Offset", minimum=-1.0, maximum=1.0, value=0.0, step=0.01)
            obj_metallic_offset = gr.Slider(label="Obj Metallic Offset", minimum=-1.0, maximum=1.0, value=0.0, step=0.01)

        with gr.Column():
            occlusion = gr.Checkbox(label='Occlusion', value=False)
            shading_maskout_mode = gr.Radio(label='Shading Maskout Mode', choices=['None', 'BBox', 'BBoxWithDepth', 'PointCloud', 'Full'], value='PointCloud')

        with gr.Column():
            shading_maskout_dilation = gr.Slider(label="Shading Maskout Bbox Dilation", minimum=0, maximum=300, value=30, step=5)
            shading_maskout_range = gr.Slider(label="Shading Maskout Range by Depth or Point Distance", minimum=0.0, maximum=10.0, value=0.6, step=0.2)

        with gr.Column():
            process_comp_btn = gr.Button(value="Process composite")

    with gr.Row():
        input_comp_depth = gr.Image(label='Comp Depth', sources=['upload'], type="numpy", width=vis_resolution)
        input_comp_normal = gr.Image(label='Comp Normal', sources=['upload'], type="numpy", width=vis_resolution)
        input_comp_shading = gr.Image(label='Comp Shading', sources=['upload'], type="numpy", width=vis_resolution)
        input_comp_mask = gr.Image(label='Comp Mask', sources=['upload'], type="numpy", width=vis_resolution)

    with gr.Row():
        input_comp_diffuse = gr.Image(label='Comp Diffuse', sources=['upload'], type="numpy", width=vis_resolution)
        input_comp_roughness = gr.Image(label='Comp Roughness', sources=['upload'], type="numpy", width=vis_resolution)
        input_comp_metallic = gr.Image(label='Comp Metallic', sources=['upload'], type="numpy", width=vis_resolution)

    with gr.Row():
        seed = gr.Slider(label="Seed", minimum=-1, maximum=2147483647, step=1, value=469)
        color_rebalance = gr.Checkbox(label='Color Balance', value=True)
        post_compositing = gr.Checkbox(label='Post Compositing', value=True)

    with gr.Row():
        generate_btn = gr.Button(value="Generate")

    with gr.Row():
        output_pred_comp = gr.Image(label='Predicted Compositing', sources=['upload'], type="numpy", height=512, width=512)
        output_pred_bg = gr.Image(label='Predicted Background', sources=['upload'], type="numpy", height=512, width=512)
        output_post_comp = gr.Image(label='Post Compositing', sources=['upload'], type="numpy", height=512, width=512)

    # with gr.Row():
    #     result_gallery = gr.Gallery(label='Composite RGB', show_label=False, elem_id="gallery")

    pred_bg_btn.click(fn=predict_bg_intrinsics,
                      inputs=[background_rgb,
                              background_dp, background_nm,
                              background_df,
                              background_rg, background_mt],
                      outputs=[input_rgb_bg, input_depth_bg, input_normal_bg, input_diffuse_bg, input_shading_bg, input_roughness_bg, input_metallic_bg])
    process_obj_btn.click(fn=process_obj_intrinsics,
                          inputs=[obj_file,
                                  obj_img, mask_img,
                                  obj_dp, obj_nm,
                                  obj_df,
                                  obj_rg, obj_mt,
                                  object_intrinsic_mode, use_rgb_as_diffuse],
                          outputs=[input_rgb_obj,
                                   input_depth_obj, input_normal_obj,
                                   input_diffuse_obj,
                                   input_roughness_obj, input_metallic_obj,
                                   input_mask_obj,
                                   input_diffuse_obj_edit])
    process_edit_btn.click(fn=process_edit_intrinsics, inputs=[input_diffuse_obj_edit])
    process_comp_btn.click(fn=process_comp,
                           inputs=[obj_relative_scale, obj_relative_vertical_position, obj_relative_horizontal_position,
                                   obj_depth_min_value, obj_depth_scale,
                                   shading_maskout_mode, shading_maskout_dilation, shading_maskout_range,
                                   occlusion,
                                   obj_diffuse_offset, obj_roughness_offset, obj_metallic_offset,
                                   ],
                           outputs=[input_comp_depth, input_comp_normal,
                                    input_comp_diffuse,
                                    input_comp_shading, input_comp_mask,
                                    input_comp_roughness, input_comp_metallic,])
    generate_btn.click(fn=generate_image, inputs=[seed, color_rebalance, post_compositing], outputs=[output_pred_comp, output_pred_bg, output_post_comp])

block.launch(server_name='127.0.0.1')
