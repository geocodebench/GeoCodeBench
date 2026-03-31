"""
Reference Implementation for apply_distortion
This serves as the ground truth for testing LLM-generated implementations.
"""

import torch
import torch.nn.functional as F
from torch import nn


def homogenize(X: torch.Tensor):
    assert X.ndim == 2
    assert X.shape[1] in (2, 3)
    return torch.cat(
        (X, torch.ones((X.shape[0], 1), dtype=X.dtype, device=X.device)), dim=1
    )


def center_crop(tensor, target_height, target_width):
    _, _, height, width = tensor.size()

    # Calculate the starting coordinates for the crop
    start_y = (height - target_height) // 2
    start_x = (width - target_width) // 2

    # Create a grid for the interpolation
    grid_y, grid_x = torch.meshgrid(torch.linspace(start_y, start_y + target_height - 1, target_height),
                                    torch.linspace(start_x, start_x + target_width - 1, target_width))
    grid = torch.stack((grid_x, grid_y), 2).unsqueeze(0).to(tensor.device)

    # Normalize grid to [-1, 1]
    grid = 2.0 * grid / torch.tensor([width - 1, height - 1]).to(tensor.device) - 1.0
    grid = grid.permute(0, 1, 2, 3).expand(tensor.size(0), target_height, target_width, 2)

    # Perform the interpolation
    cropped_tensor = F.grid_sample(tensor, grid, align_corners=True)

    return cropped_tensor


def apply_distortion(flow_apply2_gt_or_img, lens_net, P_view_insidelens_direction, P_sensor, viewpoint_cam, image, apply2gt=False, flow_scale=None):
    if flow_apply2_gt_or_img == None:
        P_view_outsidelens_direction = lens_net.forward(P_view_insidelens_direction, sensor_to_frustum=apply2gt)
        camera_directions_w_lens = homogenize(P_view_outsidelens_direction)
        control_points = camera_directions_w_lens.reshape((P_sensor.shape[0], P_sensor.shape[1], 3))[:, :, :2]

        if apply2gt:
            projection_matrix = viewpoint_cam.flow4gt
        else:
            projection_matrix = viewpoint_cam.projection_matrix
        flow = control_points @ projection_matrix[:2, :2]

    if apply2gt:
        if flow_apply2_gt_or_img == None:
            flow = nn.functional.interpolate(flow.permute(2, 0, 1).unsqueeze(0), size=(int(viewpoint_cam.image_height), int(viewpoint_cam.image_width)), mode='bilinear', align_corners=False).permute(0, 2, 3, 1).squeeze(0)
        else:
            flow = flow_apply2_gt_or_img
        gt_image = F.grid_sample(
            viewpoint_cam.fish_gt_image.unsqueeze(0),
            flow.unsqueeze(0),
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        ).squeeze(0)
        mask = (~((gt_image[0]<0.00001) & (gt_image[1]<0.00001)).unsqueeze(0)).float()
        return gt_image, mask, flow
    else:
        if flow_apply2_gt_or_img == None:
            flow = nn.functional.interpolate(flow.permute(2, 0, 1).unsqueeze(0), size=(int(viewpoint_cam.fish_gt_image_resolution[1]*flow_scale[0]), int(viewpoint_cam.fish_gt_image_resolution[2]*flow_scale[1])), mode='bilinear', align_corners=False).permute(0, 2, 3, 1).squeeze(0)
        else:
            flow = flow_apply2_gt_or_img
        image = F.grid_sample(
            image.unsqueeze(0),
            flow.unsqueeze(0),
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )
        image = center_crop(image, viewpoint_cam.fish_gt_image_resolution[1], viewpoint_cam.fish_gt_image_resolution[2]).squeeze(0)
        mask = (~((image[0]==0.0000) & (image[1]==0.0000)).unsqueeze(0)).float()
        return image, mask, flow

