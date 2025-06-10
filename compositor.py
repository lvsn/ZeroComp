import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from PIL import Image
from torchvision.transforms import v2
from torchvision.ops import masks_to_boxes
import cv2

import sdi_utils

EPS = 1e-6


def compose_bg_obj_batch(dst_batch, obj_batch,
                         conditioning_maps,
                         fill_value,
                         shading_maskout_mode='PointCloud',
                         shading_maskout_bbox_dilation=30, shading_maskout_bbox_depth_range=4.0,
                         point_cloud_fov=50, shading_maskout_pc_type='absolute', shading_maskout_pc_range=0.8, shading_maskout_pc_range_relative=1.0,
                         shading_maskout_pc_above_cropping_type='abovebbox', shading_maskout_obj_dilation=0):

    bs = obj_batch['mask'].shape[0]
    device = obj_batch['mask'].device
    image_logs = [{} for _ in range(bs)]

    obj_depth = obj_batch['depth']
    obj_mask = obj_batch['mask']

    dst_depth = dst_batch["controlnet_inputs"]["depth"]
    dst_normal = dst_batch["controlnet_inputs"]["normal"]
    dst_diffuse = dst_batch["controlnet_inputs"]["diffuse"]
    # Add a eps to background diffuse to avoid pure black diffuse
    dst_diffuse += EPS
    dst_shading = dst_batch["controlnet_inputs"]["shading"]
    if 'roughness' in conditioning_maps:
        dst_roughness = dst_batch["controlnet_inputs"]["roughness"]
    if 'metallic' in conditioning_maps:
        dst_metallic = dst_batch["controlnet_inputs"]["metallic"]

    comp_batch = {}
    for k in conditioning_maps:
        v = dst_batch["controlnet_inputs"][k].clone()
        if k == 'depth':
            dst_pil_list = sdi_utils.tensor_to_pil_list(v, [v.min(), v.max()])
        else:
            dst_pil_list = sdi_utils.tensor_to_pil_list(v)
        for sample_idx in range(bs):
            image_logs[sample_idx].update({f"dst_{k}": dst_pil_list[sample_idx]})

        tmp_mask = obj_mask.expand_as(v)
        if k == 'shading':
            shading_maskout_mode = shading_maskout_mode
            shading_maskout_bbox_dilation = shading_maskout_bbox_dilation
            shading_maskout_bbox_depth_range = shading_maskout_bbox_depth_range
            for b in range(bs):
                if shading_maskout_mode == 'None':
                    pass

                elif shading_maskout_mode == 'Obj':
                    v[b, tmp_mask[b, :, :, :] > 0.9] = fill_value

                elif 'BBox' in shading_maskout_mode:
                    # Using dilated bounding box
                    bbox = masks_to_boxes(obj_mask.squeeze(dim=1)).int()
                    _, _, h, w = v.shape

                    x1, y1, x2, y2 = bbox[b]
                    x1 = x1 - shading_maskout_bbox_dilation if x1 - shading_maskout_bbox_dilation > 0 else 0
                    y1 = y1 - shading_maskout_bbox_dilation if y1 - shading_maskout_bbox_dilation > 0 else 0
                    x2 = x2 + shading_maskout_bbox_dilation if x2 + shading_maskout_bbox_dilation < w else w
                    y2 = y2 + shading_maskout_bbox_dilation if y2 + shading_maskout_bbox_dilation < h else h

                    # Crop a rectangle in shading
                    v[b, :, y1:y2, x1:x2] = fill_value

                    if shading_maskout_mode == 'BBoxWithDepth':
                        # If higher than a threshold, use the whole source background shading
                        avg_obj_depth = obj_depth[obj_mask[b, :, :, :] > 0.9].mean()
                        bg_depth = dst_depth[b, :, :, :]
                        avg_obj_depth = avg_obj_depth.expand_as(bg_depth)
                        out_of_depth_range_mask = torch.abs(bg_depth - avg_obj_depth) > shading_maskout_bbox_depth_range

                        if shading_maskout_pc_above_cropping_type == 'abovebbox':
                            out_of_depth_range_mask[:, :y1, :] = True
                        elif shading_maskout_pc_above_cropping_type == 'argmin':
                            obj_mask_argmax = torch.argmax(obj_mask[b, :, :, :], dim=1, keepdim=True)
                            for j in range(w):
                                out_of_depth_range_mask[:, :obj_mask_argmax[0, 0, j], j] = True

                        out_of_depth_range_mask = torch.logical_and(out_of_depth_range_mask, ~(obj_mask[b, :, :, :].bool()))
                        out_of_depth_range_mask = out_of_depth_range_mask.expand_as(dst_shading[b, :, :, :])
                        v[b, out_of_depth_range_mask] = dst_shading[b, out_of_depth_range_mask]

                elif shading_maskout_mode == 'PointCloud':
                    bg_depth = dst_depth[b, :, :, :]
                    bg_point_cloud = sdi_utils.depth_map_to_point_cloud(bg_depth, fov=point_cloud_fov).permute(1, 2, 0).reshape(-1, 3)
                    obj_depth_pc = obj_depth[b, :, :, :]
                    obj_point_cloud = sdi_utils.depth_map_to_point_cloud(obj_depth_pc, fov=point_cloud_fov)
                    obj_point_cloud = obj_point_cloud.permute(1, 2, 0)[(obj_mask[b, 0, :, :] > 0.9), :]
                    dists = sdi_utils.compute_distance_bgpc_objpc(bg_point_cloud.cpu().numpy(), obj_point_cloud.cpu().numpy())
                    dists = dists.reshape(bg_depth.shape[1], bg_depth.shape[2], 1)
                    dists = torch.from_numpy(dists).to(device).permute(2, 0, 1)
                    pc_crop_mask = None
                    if shading_maskout_pc_type == 'absolute':
                        pc_crop_mask = dists < shading_maskout_pc_range
                    elif shading_maskout_pc_type == 'relative':
                        object_height = obj_point_cloud[:, 1].max() - obj_point_cloud[:, 1].min()
                        pc_crop_mask = dists < object_height * shading_maskout_pc_range_relative
                    else:
                        raise NotImplementedError

                    _, _, h, w = v.shape
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
                    v[b, crop_mask] = fill_value

                # Object mask dilation
                if shading_maskout_obj_dilation > 0:
                    ks = shading_maskout_obj_dilation
                    obj_mask_dilated = cv2.dilate(obj_mask[b, 0, :, :].cpu().numpy(), np.ones((ks, ks)), iterations=1)
                    obj_mask_dilated = torch.from_numpy(obj_mask_dilated).to(device)
                    obj_mask_dilated = obj_mask_dilated.expand_as(v[b, :, :, :])
                    v[b, obj_mask_dilated > 0.9] = fill_value

        elif k == 'depth':
            # Find the nearest valid depth value
            obj_area = obj_batch[k]
            bg_area = v

            v[tmp_mask > 0.9] = obj_area[tmp_mask > 0.9]
            v = torch.clamp(v, min=0)

        elif k == 'masked_bg':
            pass
        else:
            v[tmp_mask > 0.9] = obj_batch[k][tmp_mask > 0.9]
        # Save to comp_batch
        assert torch.isnan(v).sum() == 0, f"NaN in {k}"
        comp_batch[k] = v

    controlnet_inputs = []
    for k, v in comp_batch.items():
        if k == 'mask':
            v = torch.ones_like(v)
            shading = comp_batch['shading']
            v[shading[:, 0:1, :, :] == fill_value] = fill_value
            v = v.float()
            comp_batch[k] = v
        elif k == 'masked_bg':
            shading = comp_batch['shading']
            v[shading[:, :, :, :] == fill_value] = fill_value
            comp_batch[k] = v

            # obj_mask_b = (obj_mask > 0.9).expand_as(v)
            # v[obj_mask_b] = fill_value

        controlnet_inputs.append(v)

        if k == "depth":
            v_pil_list = sdi_utils.tensor_to_pil_list(v, [v.min(), v.max()])
        elif k == "diffuse":
            v_save = v.clone()
            v_save[obj_mask.expand_as(v) > 0.9] = v_save[obj_mask.expand_as(v) > 0.9] ** (1 / 2.2)
            v_pil_list = sdi_utils.tensor_to_pil_list(v_save.clamp_(0, 1))
        else:
            v_pil_list = sdi_utils.tensor_to_pil_list(v)

        for sample_idx in range(bs):
            image_logs[sample_idx].update({f"comp_{k}": v_pil_list[sample_idx]})

    conditioning = torch.cat(controlnet_inputs, dim=1)

    return comp_batch, conditioning, image_logs
