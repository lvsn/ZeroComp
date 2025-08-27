import logging
import math
import os
import shutil
from pathlib import Path

import comet_ml
import accelerate
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from torch.utils.data import DataLoader, ConcatDataset
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from packaging import version
from PIL import Image
from torchvision.transforms import v2
# TODO: check this
# from torchvision import transforms
from tqdm.auto import tqdm
from transformers import AutoTokenizer, PretrainedConfig

import diffusers
from diffusers import (
    AutoencoderKL,
    ControlNetModel,
    StableDiffusionControlNetPipeline,
    DDIMScheduler,
    PNDMScheduler,
    UNet2DConditionModel,
)
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available
from data.data_augmentation import RandomGammaCorrection

from data.dataset_openrooms import OpenroomsDataset
from data.dataset_openrooms_all import OpenroomsAllDataset
from data.dataset_hypersim import HypersimDataset
from data.dataset_iv import InteriorVerseDataset
from controlnet_input_handle import collate_fn, ToControlNetInput, ToPredictors, ToPredictorsWithoutEstim
from coarse_dropout import CoarseDropout

if is_wandb_available():
    import wandb

import sdi_utils
from sdi_utils import import_model_class_from_model_name_or_path
import itertools
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf
os.environ["COMET_API_KEY"] = "Dqzp3q8eKCkAUNPO6WjZSSNNC"
# hydra.output_subdir = None  # Prevent hydra from changing the working directory
# hydra.job.chdir = False  # Prevent hydra from changing the working directory

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.22.0.dev0")

logger = get_logger(__name__, "INFO")


@torch.inference_mode()
def log_validation(args, val_batch_list, to_predictors, vae, text_encoder, tokenizer, unet, controlnet, accelerator, weight_dtype, step):
    logger.info("Running validation... ")

    controlnet = accelerator.unwrap_model(controlnet)
    pipeline = StableDiffusionControlNetPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        controlnet=controlnet,
        safety_checker=None,
        revision=args.revision,
        torch_dtype=weight_dtype,
    )
    pipeline.scheduler = DDIMScheduler.from_config(
        pipeline.scheduler.config,
        **args.val_scheduler.kwargs,
    )
    # pipeline.scheduler = UniPCMultistepScheduler.from_config(pipeline.scheduler.config)
    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)

    if is_xformers_available() and args.enable_xformers_memory_efficient_attention:
        pipeline.enable_xformers_memory_efficient_attention()

    image_logs = []

    seed_count = 1

    # A 0.0 guidance scale (empty prompt) means we don't need to do 2 denoising passes!
    guidance_scale = 0.0 if args.feed_empty_prompt else 7.5

    for batch_idx, batch in enumerate(val_batch_list):
        for k, v in batch.items():
            if hasattr(v, 'to'):
                batch[k] = v.to(device=accelerator.device)
        batch = to_predictors(batch)
        controlnet_inputs_list = batch["controlnet_inputs"]
        validation_prompt = batch["caption"]
        conditioning = batch["conditioning_pixel_values"]

        for nested_seed in range(seed_count):
            if args.seed is None:
                generator = None
            else:
                generator = torch.Generator(device=accelerator.device).manual_seed(args.seed + nested_seed * 1000)

            with torch.autocast("cuda"):
                current_images = pipeline(
                    validation_prompt, conditioning, num_inference_steps=20, generator=generator, guidance_scale=guidance_scale,
                    output_type='pt'
                ).images
                current_images = torch.nan_to_num(current_images, nan=0, posinf=0, neginf=0)
                current_images = sdi_utils.tensor_to_pil_list(current_images)

            for sample_idx in range(len(current_images)):
                controlnet_inputs = {}
                for k, v in controlnet_inputs_list.items():
                    controlnet_inputs[k] = v[sample_idx]
                image_logs.append(
                    {"controlnet_inputs": controlnet_inputs, 'prompt': validation_prompt[sample_idx], "predicted_composite": current_images[sample_idx],
                     'destination_composite': batch["pixel_values"][sample_idx], 'alphabetical_id': (batch_idx * len(current_images) + sample_idx) * seed_count + nested_seed}
                )

    for tracker in accelerator.trackers:
        if tracker.name == "comet_ml":
            for sample_idx, log in enumerate(image_logs):
                controlnet_inputs_list = log["controlnet_inputs"]
                validation_prompt = log["prompt"]
                alphabetical_id = log["alphabetical_id"]
                for input_type, image in controlnet_inputs_list.items():
                    if args.random_cutout_intrinsics == True:
                        if input_type == 'depth':
                            image = sdi_utils.tensor_to_numpy(image, initial_range=(image.min(), image.max()))
                        else:
                            image = sdi_utils.tensor_to_numpy(image, initial_range=(0, 1))
                    else:
                        if input_type == 'depth':
                            depth_mask = image > 0
                            if not depth_mask.any():
                                logger.warn(f"depth image has all zero values! {alphabetical_id:04}, {input_type}")
                                continue
                            image = sdi_utils.tensor_to_numpy(image, initial_range=(image[image > 0].min(), image.max()))
                        elif 'mask' in input_type:
                            image = sdi_utils.tensor_to_numpy(image, initial_range=(0, 1))
                        elif 'normal' in input_type:
                            image = image[:3, :, :]
                            image = sdi_utils.tensor_to_numpy(image)
                        else:
                            image = sdi_utils.tensor_to_numpy(image)
                    tracker.tracker.log_image(image, name=f"s{alphabetical_id:04}, {input_type}")
                tracker.tracker.log_image(log["predicted_composite"], name=f"s{alphabetical_id:04}, prediction")
                tracker.tracker.log_image(sdi_utils.tensor_to_numpy(log["destination_composite"], initial_range=(-1, 1)), name=f"s{alphabetical_id:04}, ground truth")

        else:
            logger.warn(f"image logging not implemented for {tracker.name}")

        return image_logs


@hydra.main(config_path="configs", config_name="sdi_default", version_base='1.1')
def main(args):
    logging_dir = Path(args.output_dir, args.logging_dir)
    checkpoint_dir = Path(args.output_dir, args.checkpoint_dir)
    os.makedirs(logging_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    OmegaConf.save(args, os.path.join(args.output_dir, "config.yaml"))

    conditioning_channels = sdi_utils.get_conditioning_channels(args.conditioning_maps)
    # Handle fourier encoding for normals
    if args.fourier_encode_normals.active:
        conditioning_channels -= 3
        if args.fourier_encode_normals.include_input:
            conditioning_channels += 3 * (1 + 2 * args.fourier_encode_normals.num_freqs)
        else:
            conditioning_channels += 3 * 2 * args.fourier_encode_normals.num_freqs

    print(f"conditioning_channels: {conditioning_channels}")

    # Initialize the accelerator. We will let the accelerator handle device placement for us.
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        dynamo_backend=args.dynamo_backend,
    )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

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
    noise_scheduler = DDIMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler", **args.val_scheduler.kwargs)
    text_encoder = text_encoder_cls.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision
    )
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision)
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision
    )

    if args.controlnet_model_name_or_path:
        logger.info("Loading existing controlnet weights")
        controlnet = ControlNetModel.from_pretrained(args.controlnet_model_name_or_path)
    else:
        logger.info("Initializing controlnet weights from unet")
        controlnet = ControlNetModel.from_unet(unet, conditioning_channels=conditioning_channels)

    # `accelerate` 0.16.0 will have better support for customized saving
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):
            if accelerator.is_main_process:
                i = len(weights) - 1

                while len(weights) > 0:
                    weights.pop()
                    model = models[i]

                    sub_dir = "controlnet"
                    model.save_pretrained(os.path.join(output_dir, sub_dir))

                    i -= 1

        def load_model_hook(models, input_dir):
            while len(models) > 0:
                # pop models so that they are not loaded again
                model = models.pop()

                # load diffusers style into model
                load_model = ControlNetModel.from_pretrained(input_dir, subfolder="controlnet")
                model.register_to_config(**load_model.config)

                model.load_state_dict(load_model.state_dict())
                del load_model

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    vae.requires_grad_(False)
    unet.requires_grad_(False)
    controlnet.train()
    text_encoder.requires_grad_(False)

    # Print the number of trainable parameters and total parameters
    total_params = sum(p.numel() for p in controlnet.parameters())
    trainable_params = sum(p.numel() for p in controlnet.parameters() if p.requires_grad)
    logger.info(f"ControlNet: Total parameters: {total_params}, Trainable parameters: {trainable_params}")

    total_params = sum(p.numel() for p in unet.parameters())
    trainable_params = sum(p.numel() for p in unet.parameters() if p.requires_grad)
    logger.info(f"UNet: Total parameters: {total_params}, Trainable parameters: {trainable_params}")

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

    if args.gradient_checkpointing:
        controlnet.enable_gradient_checkpointing()

    # Check that all trainable models are in full precision
    low_precision_error_string = (
        " Please make sure to always have all model weights in full float32 precision when starting training - even if"
        " doing mixed precision training, copy of the weights should still be float32."
    )

    if accelerator.unwrap_model(controlnet).dtype != torch.float32:
        raise ValueError(
            f"Controlnet loaded as datatype {accelerator.unwrap_model(controlnet).dtype}. {low_precision_error_string}"
        )

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Use 8-bit Adam for lower memory usage or to fine-tune the model in 16GB GPUs
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
            )

        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW

    # Optimizer creation, only parameters that require gradients are optimized
    params_to_optimize = controlnet.parameters()
    optimizer = optimizer_class(
        params_to_optimize,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    to_controlnet_input = ToControlNetInput(
        device=accelerator.device,
        feed_empty_prompt=args.feed_empty_prompt,
        tokenizer=tokenizer,
        for_sdxl=False
    )

    cutout = CoarseDropout(args.aug.max_holes, args.aug.max_height, args.aug.max_width,
                           args.aug.min_holes, args.aug.min_height, args.aug.min_width,
                           args.aug.fill_value, args.aug.fill_value,
                           args.aug.always_apply, args.aug.p, args.aug.fully_drop_p,
                           args.aug.max_circles, args.aug.min_circles, args.aug.max_radius, args.aug.min_radius,
                           args.aug.p_circle)

    cutout_diffuse = None
    if args.aug_diffuse.active:
        cutout_diffuse = CoarseDropout(args.aug_diffuse.max_holes, args.aug_diffuse.max_height, args.aug_diffuse.max_width, args.aug_diffuse.p)

    train_transforms = v2.Compose([
        v2.ToImage(),
        v2.Resize(size=[args.resolution, ], interpolation=v2.InterpolationMode.BILINEAR, antialias=True),
        v2.RandomCrop([args.resolution, args.resolution]),
        # v2.RandomHorizontalFlip(p=0.5)
    ])

    color_transforms = v2.Compose([
        v2.ColorJitter(brightness=0.2),
        RandomGammaCorrection(gamma_range=(2.0, 2.25)),
    ])
    # color_transforms = None

    to_predictors = ToPredictorsWithoutEstim(accelerator.device,
                                             args.scale_destination_composite_to_minus_one_to_one,
                                             cutout,
                                             cutout_diffuse,
                                             args.aug.fill_value,
                                             args.conditioning_maps,
                                             args.inverse_cutout_mask,
                                             )
    if args.use_predictors_instead_of_gt:
        to_predictors = ToPredictors(accelerator.device,
                                     args.scale_destination_composite_to_minus_one_to_one,
                                     cutout,
                                     args.aug.fill_value,
                                     args.conditioning_maps)

    if 'openrooms' in args.dataset_name:
        if 'openrooms_all' in args.dataset_name:
            train_dataset = OpenroomsAllDataset(args.dataset_dir, 'train',
                                                transforms=train_transforms, color_transforms=color_transforms, to_controlnet_input=to_controlnet_input)
            val_dataset = OpenroomsAllDataset(args.dataset_dir, 'val',
                                              transforms=train_transforms, to_controlnet_input=to_controlnet_input)
        else:
            train_dataset = OpenroomsDataset(args.dataset_dir, 'train',
                                             transforms=train_transforms, color_transforms=color_transforms, to_controlnet_input=to_controlnet_input)
            val_dataset = OpenroomsDataset(args.dataset_dir, 'val',
                                           transforms=train_transforms, to_controlnet_input=to_controlnet_input)
    elif 'hypersim' in args.dataset_name:
        train_dataset = HypersimDataset(args.dataset_dir, 'train',
                                        transforms=train_transforms, color_transforms=color_transforms, to_controlnet_input=to_controlnet_input)
        val_dataset = HypersimDataset(args.dataset_dir, 'val',
                                      transforms=train_transforms, to_controlnet_input=to_controlnet_input)
    elif 'hybrid' in args.dataset_name:
        dataset_dir1 = args.dataset_dir.split(',')[0]
        dataset_dir2 = args.dataset_dir.split(',')[1]
        train_dataset1 = OpenroomsDataset(dataset_dir1, 'train',
                                          transforms=train_transforms, color_transforms=color_transforms, to_controlnet_input=to_controlnet_input)
        val_dataset1 = OpenroomsDataset(dataset_dir1, 'val',
                                        transforms=train_transforms, to_controlnet_input=to_controlnet_input)
        train_dataset2 = HypersimDataset(dataset_dir2, 'train',
                                         transforms=train_transforms, color_transforms=color_transforms, to_controlnet_input=to_controlnet_input)
        val_dataset2 = HypersimDataset(dataset_dir2, 'val',
                                       transforms=train_transforms, to_controlnet_input=to_controlnet_input)
        train_dataset = ConcatDataset([train_dataset1, train_dataset2])
        val_dataset = ConcatDataset([val_dataset1, val_dataset2])
    elif 'interior_verse' in args.dataset_name:
        train_dataset = InteriorVerseDataset(args.dataset_dir, 'train',
                                             transforms=train_transforms, color_transforms=color_transforms, to_controlnet_input=to_controlnet_input)
        val_dataset = InteriorVerseDataset(args.dataset_dir, 'val',
                                           transforms=train_transforms, to_controlnet_input=to_controlnet_input)
    else:
        raise ValueError(f"Unknown dataset {args.dataset_dir}")

    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
        collate_fn=collate_fn
    )
    val_dataloader = DataLoader(
        val_dataset,
        # shuffle=False,
        shuffle=True,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
        collate_fn=collate_fn,
        drop_last=True
    )
    val_loader_iter = iter(val_dataloader)

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )

    # Prepare everything with our `accelerator`.
    controlnet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        controlnet, optimizer, train_dataloader, lr_scheduler
    )

    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move vae, unet and text_encoder to device and cast to weight_dtype
    vae.to(accelerator.device, dtype=weight_dtype)
    unet.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        from copy import copy
        tracker_config = dict(copy(args))

        accelerator.init_trackers(args.tracker_project_name, config=tracker_config)

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            # path = os.path.basename(args.resume_from_checkpoint)
            path = args.resume_from_checkpoint
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(checkpoint_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            if args.resume_from_checkpoint != "latest":
                accelerator.load_state(path)
            else:
                accelerator.load_state(os.path.join(checkpoint_dir, path))
            global_step = int(path.split("-")[-1].strip('/'))

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch
    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    for epoch in range(first_epoch, args.num_train_epochs):
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(controlnet):
                # Convert input to controlnet format
                batch = to_predictors(batch)
                name = batch['name']

                # Check for NaNs
                # controlnet_inputs = batch['controlnet_inputs']
                # for k, v in controlnet_inputs.items():
                #     assert not torch.isnan(v).any(), f"{k} is nan! {name}"

                # Detect if there's a pure black image in the batch, for openrooms_all
                # if 'openrooms_all' in args.dataset_name:
                #     mean_values = batch["pixel_values"].mean(dim=(1, 2, 3))
                #     black_images = mean_values < 0.01
                #     if black_images.any():
                #         logger.warn(f"Black image detected in the batch! {name}")
                #         global_step += 1
                #         continue

                # Convert images to latent space
                latents = vae.encode(batch["pixel_values"].to(dtype=weight_dtype)).latent_dist.sample()
                latents = latents * vae.config.scaling_factor
                # assert not torch.isnan(latents).any(), f"latent is nan! {name}"

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # Get the text embedding for conditioning
                input_ids = batch["input_ids"]
                encoder_hidden_states = text_encoder(input_ids)[0]

                controlnet_image = batch["conditioning_pixel_values"].to(dtype=weight_dtype)

                # color_cond = torch.rand_like(timesteps)

                down_block_res_samples, mid_block_res_sample = controlnet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=encoder_hidden_states,
                    controlnet_cond=controlnet_image,
                    return_dict=False,
                    # timestep_cond=shading_avgcolor_emb
                )

                # Predict the noise residual
                model_pred = unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=encoder_hidden_states,
                    down_block_additional_residuals=[
                        sample.to(dtype=weight_dtype) for sample in down_block_res_samples
                    ],
                    mid_block_additional_residual=mid_block_res_sample.to(dtype=weight_dtype),
                ).sample

                # assert not torch.isnan(model_pred).any(), f"model_pred is nan! {name}"

                # Get the target for loss depending on the prediction type
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                # Add mask to loss, when depth/diffuse==0, we don't backpropagate
                if 'depth_valid_mask' in batch['controlnet_inputs'] and args.valid_mask_type == 'depth_valid_mask_loss':
                    valid_mask = batch['controlnet_inputs']['depth_valid_mask'].to(dtype=weight_dtype)
                    valid_mask = F.adaptive_avg_pool2d(valid_mask, model_pred.shape[-2:])
                    valid_mask = valid_mask.expand_as(model_pred)
                    valid_mask = valid_mask > 0.9
                    model_pred = model_pred[valid_mask]
                    target = target[valid_mask]

                # Original loss
                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

                if torch.isnan(loss).any():
                    logger.warn(f"loss is nan! skipping this iteration {global_step}. {name}")
                    global_step += 1
                    continue

                # assert not torch.isnan(loss).any(), f"loss is nan! {name}"

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    params_to_clip = controlnet.parameters()
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=args.set_grads_to_none)

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process:
                    if global_step % args.checkpointing_steps == 0:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(checkpoint_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(checkpoint_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(checkpoint_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

                    if global_step % args.validation_steps == 0 or global_step == 1:
                        val_batch_list = []
                        for _ in range(args.val_batch_size):
                            try:
                                val_batch = next(val_loader_iter)
                            except StopIteration:
                                val_loader_iter = iter(val_dataloader)
                                val_batch = next(val_loader_iter)
                            val_batch_list.append(val_batch)
                        log_validation(
                            args,
                            val_batch_list,
                            to_predictors,
                            vae,
                            text_encoder,
                            tokenizer,
                            unet,
                            controlnet,
                            accelerator,
                            weight_dtype,
                            global_step,
                        )
            logs = {"loss": loss.detach().item(),
                    "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break

    # Create the pipeline using using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        controlnet = accelerator.unwrap_model(controlnet)
        controlnet.save_pretrained(checkpoint_dir)

    accelerator.end_training()


if __name__ == "__main__":
    main()
