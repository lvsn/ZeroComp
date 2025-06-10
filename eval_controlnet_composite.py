import logging
import os

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from torch.utils.data import DataLoader
# from accelerate.logging import get_logger
import logging
from packaging import version
from PIL import Image
from torchvision.transforms import v2
from torchvision.ops import masks_to_boxes
from tqdm.auto import tqdm
from transformers import AutoTokenizer, PretrainedConfig
from torchvision.utils import make_grid
from diffusers.utils.torch_utils import randn_tensor

from diffusers import (
    AutoencoderKL,
    AsymmetricAutoencoderKL,
    ControlNetModel,
    StableDiffusionControlNetPipeline,
    DDIMScheduler,
    UNet2DConditionModel,
)
from diffusers.utils import check_min_version, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available

from data.dataset_labo import LaboDataset
from controlnet_input_handle import ToControlNetInput, ToPredictors, collate_fn, match_depth_from_footprint

from compositor import compose_bg_obj_batch

import sdi_utils
from sdi_utils import import_model_class_from_model_name_or_path
import hydra
from omegaconf import OmegaConf
# hydra.output_subdir = None  # Prevent hydra from changing the working directory
# hydra.job.chdir = False  # Prevent hydra from changing the working directory

EPS = 1e-6

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.22.0.dev0")

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@hydra.main(config_path="configs", config_name="sdi_default", version_base='1.1')
def main(args):
    # Seed everything
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    conditioning_channels = sdi_utils.get_conditioning_channels(args.conditioning_maps)

    if 'openrooms' in args.eval.controlnet_model_name_or_path:
        training_dataset = 'openrooms'
    elif 'hypersim' in args.eval.controlnet_model_name_or_path:
        training_dataset = 'hypersim'
    elif 'iv' or 'interior_verse' in args.eval.controlnet_model_name_or_path:
        training_dataset = 'iv'

    results_dir = f"{args.eval.results_dir}-{training_dataset}-{args.eval.shading_maskout_mode}"
    os.makedirs(results_dir, exist_ok=True)
    OmegaConf.save(args, os.path.join(os.path.split(results_dir)[-2], f'{os.path.split(results_dir)[-1]}.yaml'))

    # Load the tokenizer
    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, revision=args.revision, use_fast=False)
    elif args.pretrained_model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="tokenizer",
            revision=args.revision,
            use_fast=False,
        )

    # import correct text encoder class
    text_encoder_cls = import_model_class_from_model_name_or_path(args.pretrained_model_name_or_path, args.revision)

    # Load scheduler and models
    text_encoder = text_encoder_cls.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision
    )
    if args.vae_type == 'normal':
        vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision)
    elif args.vae_type == 'asymmetric':
        vae = AsymmetricAutoencoderKL.from_pretrained(args.vae_name_or_path)
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision
    )

    logger.info("Loading existing controlnet weights")
    controlnet = ControlNetModel.from_pretrained(args.eval.controlnet_model_name_or_path, subfolder="controlnet")
    # load_model = ControlNetModel.from_pretrained(args.eval.controlnet_model_name_or_path, subfolder="controlnet")
    # model.register_to_config(**load_model.config)

    # model.load_state_dict(load_model.state_dict())
    # del load_model

    vae.requires_grad_(False)
    unet.requires_grad_(False)
    text_encoder.requires_grad_(False)
    controlnet.requires_grad_(False)
    controlnet.eval()

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warn(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            unet.enable_xformers_memory_efficient_attention()
            controlnet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    to_controlnet_input = ToControlNetInput(
        device=args.eval.device,
        feed_empty_prompt=args.feed_empty_prompt,
        tokenizer=tokenizer,
        for_sdxl=False
    )

    aug_cutout = None

    to_predictors = ToPredictors(args.eval.device,
                                 args.scale_destination_composite_to_minus_one_to_one,
                                 aug_cutout,
                                 conditioning_maps=['depth', 'normal', 'diffuse', 'mask', 'shading', 'roughness', 'metallic'],
                                 predictor_names=args.eval.predictor_names,)

    val_transforms = v2.Compose([
        v2.ToImage(),
        v2.Resize(size=[args.resolution, ], antialias=True),
        v2.CenterCrop([args.resolution, args.resolution])
    ])

    if args.dataset_name == 'labo':
        val_dataset = LaboDataset(args.dataset_dir,
                                  transforms=val_transforms, to_controlnet_input=to_controlnet_input)
    else:
        raise ValueError(f"Unknown dataset name: {args.dataset}")

    val_dataloader = DataLoader(
        val_dataset,
        shuffle=False,
        batch_size=args.eval.eval_batch_size,
        num_workers=args.dataloader_num_workers,
        collate_fn=collate_fn
    )

    # Move vae, unet and text_encoder to device
    weight_dtype = torch.float32
    if args.eval.weight_dtype == "fp16":
        weight_dtype = torch.float16
    elif args.eval.weight_dtype == "bf16":
        weight_dtype = torch.bfloat16
    vae.to(args.eval.device, dtype=weight_dtype)
    unet.to(args.eval.device, dtype=weight_dtype)
    text_encoder.to(args.eval.device, dtype=weight_dtype)
    controlnet.to(args.eval.device, dtype=weight_dtype)

    pipeline = StableDiffusionControlNetPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        controlnet=controlnet,
        safety_checker=None,
        revision=args.revision
    )
    pipeline.scheduler = DDIMScheduler.from_config(
        pipeline.scheduler.config,
        **args.val_scheduler.kwargs,
    )
    pipeline = pipeline.to(args.eval.device, dtype=weight_dtype)
    pipeline.set_progress_bar_config(disable=True)

    if args.enable_xformers_memory_efficient_attention:
        pipeline.enable_xformers_memory_efficient_attention()

    for batch_idx, batch in enumerate(tqdm(val_dataloader)):
        with torch.inference_mode():
            bs = batch['pixel_values'].shape[0]
            image_logs = [{} for _ in range(bs)]

            for k, v in batch.items():
                if hasattr(v, 'to'):
                    batch[k] = v.to(device=args.eval.device)

            obj_mask = batch["mask"]
            # Get object intrinsics directly from dataset
            obj_batch = {
                "depth": batch["depth"] if "depth" in batch else None,
                "footprint_depth": batch["footprint_depth"] if "footprint_depth" in batch else None,
                "normal": batch["normal"],
                "diffuse": batch["diffuse"],
                "mask": batch["mask"],
            }
            if args.eval.use_rgb_as_diffuse:
                obj_batch["diffuse"] = batch["src_obj"]
            if 'roughness' in args.conditioning_maps:
                obj_batch['roughness'] = torch.clip(batch['roughness'] + args.eval.roughness_offset, 0, 1)
            if 'metallic' in args.conditioning_maps:
                obj_batch['metallic'] = torch.clip(batch['metallic'] + args.eval.metallic_offset, 0, 1)
            if args.eval.albedo_gamma_correction:
                obj_batch['diffuse'] = obj_batch['diffuse'] ** (1 / 2.2)
            obj_depth = obj_batch['depth']

            # Get object intrinsics from rgb image
            # src_batch = {'pixel_values': batch['src_comp'],
            #              'input_ids': batch['input_ids'],
            #              'caption': batch['caption']}
            # src_batch = to_predictors(src_batch)
            # obj_batch = src_batch["controlnet_inputs"]

            # Get background intrinsics
            dst_batch = {
                "pixel_values": batch["pixel_values"],
                "input_ids": batch["input_ids"],
                "caption": batch["caption"],
                "bg_depth": batch["bg_depth"] if "bg_depth" in batch else None,
                "bg_normal": batch["bg_normal"] if "bg_normal" in batch else None,
                "bg_diffuse": batch["bg_diffuse"] if "bg_diffuse" in batch else None,
                "bg_roughness": batch["bg_roughness"] if "bg_roughness" in batch else None,
                "bg_metallic": batch["bg_metallic"] if "bg_metallic" in batch else None,
            }
            dst_batch = to_predictors(dst_batch)
            dst_depth = dst_batch["controlnet_inputs"]["depth"]
            dst_normal = dst_batch["controlnet_inputs"]["normal"]
            dst_diffuse = dst_batch["controlnet_inputs"]["diffuse"]
            # Add a eps to background diffuse to avoid pure black diffuse
            dst_diffuse += EPS
            dst_shading = dst_batch["controlnet_inputs"]["shading"]
            if 'roughness' in args.conditioning_maps:
                dst_roughness = dst_batch["controlnet_inputs"]["roughness"]
            if 'metallic' in args.conditioning_maps:
                dst_metallic = dst_batch["controlnet_inputs"]["metallic"]

            # Log src and obj images
            src_obj_list = sdi_utils.tensor_to_pil_list(batch["src_obj"])
            # src_comp_list = sdi_utils.tensor_to_pil_list(batch["src_comp"])
            dst_comp_list = sdi_utils.tensor_to_pil_list(batch["comp"])
            bg_list = sdi_utils.tensor_to_pil_list(dst_batch["pixel_values"] * 0.5 + 0.5)
            bg_image_for_balance = dst_batch["pixel_values"].clone() * 0.5 + 0.5
            for sample_idx in range(bs):
                image_logs[sample_idx].update({"src_obj": src_obj_list[sample_idx],
                                               #    "src_comp": src_comp_list[sample_idx],
                                               "src_mask": obj_mask[sample_idx],
                                               "dst_bg": bg_list[sample_idx],
                                               "dst_comp": dst_comp_list[sample_idx],
                                               "name": batch["name"][sample_idx]} if "name" in batch else False)

            if 'depth' in args.conditioning_maps:
                if obj_batch['footprint_depth'] is not None:
                    if args.eval.adjust_bgdepth_to_objdepth:
                        dst_depth = match_depth_from_footprint(dst_depth, obj_batch['depth'], obj_batch['footprint_depth'], args.eval.adjust_bgdepth_to_objdepth)
                        dst_batch["controlnet_inputs"]["depth"] = dst_depth.clone()
                        dst_depth = match_depth_from_footprint(dst_depth, obj_depth,
                                                               obj_batch['footprint_depth'], args.eval.adjust_bgdepth_to_objdepth)
                    else:
                        obj_batch['depth'] = match_depth_from_footprint(dst_depth, obj_batch['depth'], obj_batch['footprint_depth'], args.eval.adjust_bgdepth_to_objdepth)
                        obj_depth = match_depth_from_footprint(dst_depth, obj_depth,
                                                               obj_batch['footprint_depth'], args.eval.adjust_bgdepth_to_objdepth)
                else:
                    logger.warning(
                        f"Footprint depth is not provided for {batch['name']}, using the smallest bg depth value inside the object mask as the minimum object depth."
                    )
                    # When no footprint depth is provided, we find the nearest valid bg depth value and let it be the minimum object depth
                    tmp_mask = obj_batch['mask']
                    obj_depth = obj_batch["depth"].clone()
                    obj_area = obj_depth[tmp_mask > 0.9]
                    min_obj_depth = obj_area.min()
                    # max_obj_depth = obj_area.max()

                    bg_depth = dst_batch["controlnet_inputs"]["depth"].clone()
                    bg_area = bg_depth[tmp_mask > 0.9]
                    min_bg_depth = bg_area.min()
                    # max_bg_depth = bg_area.max()
                    # avg_bg_depth = bg_area.mean()

                    obj_depth = (obj_depth - min_obj_depth) + min_bg_depth
                    obj_batch['depth'] = obj_depth

            # Compose the background and object
            comp_batch, conditioning, comp_image_logs = compose_bg_obj_batch(dst_batch, obj_batch,
                                                                             args.conditioning_maps,
                                                                             args.aug.fill_value,
                                                                             args.eval.shading_maskout_mode,
                                                                             args.eval.shading_maskout_bbox_dilation, args.eval.shading_maskout_bbox_depth_range,
                                                                             args.eval.point_cloud_fov, args.eval.shading_maskout_pc_type, args.eval.shading_maskout_pc_range, args.eval.shading_maskout_pc_range_relative,
                                                                             args.eval.shading_maskout_pc_above_cropping_type, args.eval.shading_maskout_obj_dilation)

            image_logs = [dict(list(image_logs[i].items()) + list(comp_image_logs[i].items())) for i in range(bs)]

            validation_prompt = dst_batch["caption"]

            # Get compositing from different seeds
            seed_count = 1
            for nested_seed in range(seed_count):
                if args.seed is None:
                    generator = None
                else:
                    generator = torch.Generator(device=args.eval.device).manual_seed(args.seed + nested_seed * 1000)

                with torch.autocast("cuda"):
                    noise_latents = randn_tensor([bs, pipeline.unet.config.in_channels, args.resolution // pipeline.vae_scale_factor, args.resolution // pipeline.vae_scale_factor],
                                                 generator=generator, device=pipeline.device, dtype=pipeline.dtype)

                    if args.vae_type == "asymmetric":
                        current_latent = pipeline(
                            validation_prompt, conditioning, num_inference_steps=20, generator=generator, guidance_scale=0,
                            output_type='latent'
                        ).images

                        mask_image = comp_batch["mask"]
                        mask_image[comp_batch["mask"] < 0.1] = 1
                        mask_image[comp_batch["mask"] > 0.9] = 0
                        # mask_image_save = sdi_utils.tensor_to_pil(mask_image)
                        # mask_image_save.save(os.path.join('tmp', f"{batch_idx:05}_mask_image.png"))
                        bg_image = dst_batch["pixel_values"]
                        # bg_image = comp_batch["masked_bg"]
                        bg_image[comp_batch["mask"].expand_as(bg_image) == args.aug.fill_value] = 0
                        # bg_image_save = sdi_utils.tensor_to_pil(bg_image)
                        # bg_image_save.save(os.path.join('tmp', f"{batch_idx:05}_bg_image.png"))

                        current_images = vae.decode(current_latent, mask=mask_image, image=bg_image).sample
                        current_images = (current_images * 0.5 + 0.5).clamp(0, 1)
                    else:
                        current_images = pipeline(
                            validation_prompt, conditioning, num_inference_steps=20, generator=generator, latents=noise_latents, guidance_scale=0,
                            output_type='pt'
                        ).images

                    current_images = torch.nan_to_num(current_images, nan=0, posinf=0, neginf=0)

                    if args.eval.post_compositing:
                        dst_bg = dst_batch["pixel_values"]
                        dst_bg = (dst_bg * 0.5 + 0.5).clamp(0, 1)

                        dst_mask = torch.ones([bs, 1, args.resolution, args.resolution], device=dst_bg.device, dtype=dst_bg.dtype)

                        # Rebuttal
                        # Also crop the dst_shading according to the comp_mask
                        dst_mask = comp_batch["mask"]
                        dst_shading_mask = comp_batch["mask"].expand_as(dst_shading)
                        dst_shading[dst_shading_mask < 0.1] = args.aug.fill_value
                        for sample_idx in range(bs):
                            image_logs[sample_idx].update({f"dst_shading": sdi_utils.tensor_to_pil(dst_shading[sample_idx].clone())})
                            image_logs[sample_idx].update({f"dst_mask": sdi_utils.tensor_to_pil(dst_mask[sample_idx].clone())})

                        dst_conditioning_list = []
                        for map_name in args.conditioning_maps:
                            if map_name == "depth":
                                dst_conditioning_list.append(dst_depth)
                            elif map_name == "normal":
                                dst_conditioning_list.append(dst_normal)
                            elif map_name == "diffuse":
                                dst_conditioning_list.append(dst_diffuse)
                            elif map_name == "shading":
                                dst_conditioning_list.append(dst_shading)
                            elif map_name == "mask":
                                dst_conditioning_list.append(dst_mask)
                            elif map_name == "roughness":
                                dst_conditioning_list.append(dst_roughness)
                            elif map_name == "metallic":
                                dst_conditioning_list.append(dst_metallic)
                            elif map_name == "bg":
                                dst_conditioning_list.append(dst_bg)

                        dst_conditioning = torch.cat(dst_conditioning_list, dim=1)
                        dst_images = pipeline(
                            validation_prompt, dst_conditioning, num_inference_steps=20, generator=generator, latents=noise_latents, guidance_scale=0,
                            output_type='pt'
                        ).images

                        dst_images = torch.nan_to_num(dst_images, nan=0, posinf=0, neginf=0)

                        # Composite it back to original background image
                        # Get the shadow area without the object
                        comp_mask = comp_batch["mask"]
                        # Calculate the visibility of the shadow area
                        intensity_from_rgb = torch.tensor([0.299, 0.587, 0.114], device=args.eval.device).view(1, 3, 1, 1)
                        current_images_intensity = current_images * intensity_from_rgb
                        dst_images_intensity = dst_images * intensity_from_rgb
                        visibility = (current_images_intensity.sum(dim=1, keepdim=True) + EPS) / (dst_images_intensity.sum(dim=1, keepdim=True) + EPS)
                        visibility = visibility.clamp(0, 1)

                        visibility_mask = comp_mask.clamp(0, 1)
                        visibility_mask = 1 - visibility_mask
                        visibility_mask = v2.functional.gaussian_blur(visibility_mask, (15, 15), 1.5)
                        visibility = visibility * visibility_mask + 1 * (1 - visibility_mask)

                        # Dilate object mask
                        # obj_mask_dilated = cv2.dilate(obj_mask[0, 0, :, :].cpu().numpy(), np.ones((2, 2)), iterations=1)
                        # obj_mask_dilated = torch.from_numpy(obj_mask_dilated).to(args.eval.device).unsqueeze(0).unsqueeze(0)
                        # obj_mask_dilated = obj_mask_dilated.expand_as(compositing)
                        # obj_mask_dilated_save = Image.fromarray((obj_mask_dilated[0, 0, :, :].float().cpu().numpy() * 255).astype(np.uint8))
                        # obj_mask_dilated_save.save(os.path.join('tmp', f"{batch_idx:05}_obj_mask_dilated.png"))
                        # obj_mask_filtered = cv2.GaussianBlur(obj_mask[0, 0, :, :].cpu().numpy(), (5, 5), 0)
                        # obj_mask_filtered = torch.from_numpy(obj_mask_filtered).to(args.eval.device).unsqueeze(0).unsqueeze(0)
                        # obj_mask_filtered = v2.functional.gaussian_blur(obj_mask, (9, 9), 1)
                        obj_mask_filtered = obj_mask

                        # Rebuttal
                        # Do the color balance for the object
                        if args.eval.obj_color_balance:
                            # current_images_balanced = current_images * sdi_utils.color_rebalance(current_images, bg_image_for_balance)  # Buggy one
                            current_images_balanced = current_images * sdi_utils.color_rebalance(dst_images, bg_image_for_balance)  # Correct one
                        else:
                            current_images_balanced = current_images
                        compositing = dst_bg * visibility * (1 - obj_mask_filtered) + current_images_balanced * obj_mask_filtered

                for sample_idx in range(len(current_images)):
                    image_logs[sample_idx][f"pred_seed_{seed_count}"] = current_images[sample_idx]
                    image_logs[sample_idx]['obj_mask'] = obj_mask[sample_idx]
                    # image_logs[sample_idx][f"pred_seed_{seed_count}_balanced"] = current_images_balanced[sample_idx]
                    if args.eval.post_compositing:
                        image_logs[sample_idx][f"pred_seed_{seed_count}_comp"] = compositing[sample_idx]
                        image_logs[sample_idx][f"pred_seed_{seed_count}_bg"] = dst_images[sample_idx]
                        # image_logs[sample_idx][f'pred_seed_{seed_count}_bg_balanced'] = dst_images_balanced[sample_idx]
                        image_logs[sample_idx]['visibility'] = visibility[sample_idx]
                        image_logs[sample_idx]['visibility_mask'] = visibility_mask[sample_idx]
                        if conditioning_channels == 14:
                            image_logs[sample_idx][f"masked_bg"] = comp_batch["masked_bg"][sample_idx]

            for sample_idx, log in enumerate(image_logs):
                id = batch_idx * len(current_images) + sample_idx

                # Create grid image
                comp_image_names = [f"comp_{k}" for k in args.conditioning_maps]
                dst_image_names = ['src_obj', 'dst_bg', 'obj_mask', 'visibility' if args.eval.post_compositing else 'blank']
                pred_image_names = [f'pred_seed_{seed_count}_bg', f'pred_seed_{seed_count}', f'pred_seed_{seed_count}_comp',
                                    'dst_comp'] if args.eval.post_compositing else [f'pred_seed_{seed_count}', 'dst_comp']
                row_num = max(len(comp_image_names), len(dst_image_names), len(pred_image_names))
                if len(comp_image_names) < row_num:
                    comp_image_names += ['blank'] * (row_num - len(comp_image_names))
                if len(dst_image_names) < row_num:
                    dst_image_names += ['blank'] * (row_num - len(dst_image_names))
                if len(pred_image_names) < row_num:
                    pred_image_names += ['blank'] * (row_num - len(pred_image_names))
                grid_image_names = comp_image_names + dst_image_names + pred_image_names

                grid_image_dict = {}
                for k, v in log.items():
                    if k not in grid_image_names:
                        continue
                    if isinstance(v, torch.Tensor):
                        v = v.cpu()
                    elif isinstance(v, Image.Image):
                        # transform to tensor
                        v = np.asarray(v, dtype=np.float32) / 255.0
                        if len(v.shape) == 2:
                            v = np.expand_dims(v, axis=2)
                        v = torch.from_numpy(v).permute(2, 0, 1)
                    else:
                        raise ValueError(f"Unknown type {type(v)}")
                    if v.shape[0] == 1:
                        v = v.expand(3, -1, -1)
                    if len(v.shape) == 4:
                        v = v.squeeze(dim=0)
                    grid_image_dict[k] = v
                grid_image_dict['blank'] = torch.ones([3, args.resolution, args.resolution], dtype=torch.float32) * 0.0

                grid_image = make_grid([grid_image_dict[k] for k in grid_image_names], nrow=len(grid_image_names) // 3)
                # ezexr.imwrite(os.path.join(results_dir, f"{id:05}.exr"), grid_image.squeeze().permute(1, 2, 0).numpy())
                grid_image = sdi_utils.tensor_to_pil(grid_image)
                if log['name']:
                    grid_image.save(os.path.join(results_dir, f"{log['name']}_grid.jpg"), quality=90)
                else:
                    grid_image.save(os.path.join(results_dir, f"{id:05}_grid.jpg"), quality=90)

                if args.eval.output_all:
                    for k, v in log.items():
                        if isinstance(v, torch.Tensor):
                            v = sdi_utils.tensor_to_pil(v.to(device="cpu"))
                        elif isinstance(v, Image.Image):
                            pass
                        elif isinstance(v, str):
                            continue
                        else:
                            raise ValueError(f"Unknown type {type(v)}")

                        if log['name']:
                            v.save(os.path.join(results_dir, f"{log['name']}_{k}.png"))
                        else:
                            v.save(os.path.join(results_dir, f"{id:05}_{k}.png"))
                else:
                    log_filtered = {}
                    if args.eval.post_compositing:
                        log_filtered['pred_seed_1_comp'] = log['pred_seed_1_comp']
                    else:
                        log_filtered['pred_seed_1'] = log['pred_seed_1']
                    # log_filtered['comp_depth'] = log['comp_depth']
                    # log_filtered['comp_normal'] = log['comp_normal']
                    # log_filtered['comp_diffuse'] = log['comp_diffuse']
                    # log_filtered['comp_shading'] = log['comp_shading']

                    for k, v in log_filtered.items():
                        if isinstance(v, torch.Tensor):
                            v = sdi_utils.tensor_to_pil(v.to(device="cpu"))
                        elif isinstance(v, Image.Image):
                            pass
                        elif isinstance(v, str):
                            continue
                        else:
                            raise ValueError(f"Unknown type {type(v)}")

                        if log['name']:
                            v.save(os.path.join(results_dir, f"{log['name']}_{k}.png"))
                        else:
                            v.save(os.path.join(results_dir, f"{id:05}_{k}.png"))


if __name__ == "__main__":
    main()
