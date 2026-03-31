#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import math
import numpy as np
from typing import NamedTuple
import torch.nn.functional as F


def fibonacci_sphere_sampling(normals, sample_num, random_rotate=True):
    pre_shape = normals.shape[:-1]
    if len(pre_shape) > 1:
        normals = normals.reshape(-1, 3)
    delta = np.pi * (3.0 - np.sqrt(5.0))

    # fibonacci sphere sample around z axis
    idx = torch.arange(sample_num, dtype=torch.float, device='cuda')[None]
    z = (1 - 2 * idx / (2 * sample_num - 1)).clamp_min(np.sin(10/180*np.pi))
    rad = torch.sqrt(1 - z ** 2)
    theta = delta * idx
    if random_rotate:
        theta = torch.rand(*pre_shape, 1, device='cuda') * 2 * np.pi + theta
    y = torch.cos(theta) * rad
    x = torch.sin(theta) * rad
    z_samples = torch.stack([x, y, z.expand_as(y)], dim=-2)

    # rotate to normal
    # z_vector = torch.zeros_like(normals)
    # z_vector[..., 2] = 1  # [H, W, 3]
    # rotation_matrix = rotation_between_vectors(z_vector, normals)
    rotation_matrix = rotation_between_z(normals)
    incident_dirs = rotation_matrix @ z_samples
    incident_dirs = F.normalize(incident_dirs, dim=-2).transpose(-1, -2)
    incident_areas = torch.ones_like(incident_dirs)[..., 0:1] * 2 * np.pi
    if len(pre_shape) > 1:
        incident_dirs = incident_dirs.reshape(*pre_shape, sample_num, 3)
        incident_areas = incident_areas.reshape(*pre_shape, sample_num, 1)
    return incident_dirs, incident_areas

class BasicPointCloud(NamedTuple):
    points : np.array
    colors : np.array
    normals : np.array

def geom_transform_points(points, transf_matrix):
    P, _ = points.shape
    ones = torch.ones(P, 1, dtype=points.dtype, device=points.device)
    points_hom = torch.cat([points, ones], dim=1)
    points_out = torch.matmul(points_hom, transf_matrix.unsqueeze(0))

    denom = points_out[..., 3:] + 0.0000001
    return (points_out[..., :3] / denom).squeeze(dim=0)

def getWorld2View(R, t):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0
    return np.float32(Rt)

def getWorld2View2(R, t, translate=np.array([.0, .0, .0]), scale=1.0):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    C2W = np.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = np.linalg.inv(C2W)
    return np.float32(Rt)

def getProjectionMatrix(znear, zfar, fovX, fovY):
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros(4, 4)

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P

# #
def getProjectionMatrixCorrect(znear, zfar, H, W, K):

  top = (K[1,2])/K[1,1] * znear
  bottom = -(H - K[1,2])/K[1,1] * znear
  right = (K[0,2])/K[0,0] * znear
  left = -(W - K[0,2])/K[0,0] * znear

  P = torch.zeros(4, 4)

  z_sign = 1.0

  P[0, 0] = 2.0 * znear / (right - left)
  P[1, 1] = 2.0 * znear / (top - bottom)
  P[0, 2] = (right + left) / (right - left)
  P[1, 2] = (top + bottom) / (top - bottom)
  P[3, 2] = z_sign
  P[2, 2] = z_sign * zfar / (zfar - znear)
  P[2, 3] = -(zfar * znear) / (zfar - znear)
  return P


def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))

def focal2fov(focal, pixels):
    return 2*math.atan(pixels/(2*focal))

def rotation_between_z(vec):
    """
    Args:
        vec: [..., 3]

    Returns:
        R: [..., 3, 3]

    """
    ****EMPTY****
    return R

def rgb_to_srgb(img, clip=True):
    # hdr img
    if isinstance(img, np.ndarray):
        # assert len(img.shape) == 3, img.shape
        # assert img.shape[2] == 3, img.shape
        img = np.where(img > 0.0031308, np.power(np.maximum(img, 0.0031308), 1.0 / 2.4) * 1.055 - 0.055, 12.92 * img)
        if clip:
            img = np.clip(img, 0.0, 1.0)
        return img
    elif isinstance(img, torch.Tensor):
        # assert len(img.shape) == 3, img.shape
        # assert img.shape[0] == 3, img.shape
        img = torch.where(img > 0.0031308, torch.pow(torch.max(img, torch.tensor(0.0031308)), 1.0 / 2.4) * 1.055 - 0.055, 12.92 * img)
        if clip:
            img = torch.clamp(img, 0.0, 1.0)
        return img
    else:
        raise TypeError("Unsupported input type. Supported types are numpy.ndarray and torch.Tensor.")


def srgb_to_rgb(img):
    # f is LDR
    if isinstance(img, np.ndarray):
        img = np.where(img <= 0.04045, img / 12.92, np.power((np.maximum(img, 0.04045) + 0.055) / 1.055, 2.4))
        return img
    elif isinstance(img, torch.Tensor):
        img = torch.where(img <= 0.04045, img / 12.92, torch.pow((torch.max(img, torch.tensor(0.04045)) + 0.055) / 1.055, 2.4))
        return img
    else:
        raise TypeError("Unsupported input type. Supported types are numpy.ndarray and torch.Tensor.")