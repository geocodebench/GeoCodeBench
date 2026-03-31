"""
Reference Implementation for sphere2pose function
This is the correct implementation from pvd_utils.py
"""

import torch
import copy


def sphere2pose(c2ws_input, theta, phi, r, device, x=None, y=None):
    """
    Transform camera poses using spherical coordinates.
    
    Args:
        c2ws_input: Input camera-to-world transformation matrices, shape (batch_size, 4, 4)
        theta: Rotation angle around X-axis (in degrees)
        phi: Rotation angle around Y-axis (in degrees)
        r: Translation along Z-axis
        device: Device to run computations on (e.g., 'cpu', 'cuda')
        x: Optional translation along X-axis
        y: Optional translation along Y-axis
    
    Returns:
        c2ws: Transformed camera-to-world matrices, shape (batch_size, 4, 4)
    """
    c2ws = copy.deepcopy(c2ws_input)

    # First translate along world coordinate system z-axis, then rotate
    c2ws[:, 2, 3] += r
    if x is not None:
        c2ws[:, 1, 3] += y
    if y is not None:
        c2ws[:, 0, 3] += x

    theta = torch.deg2rad(torch.tensor(theta)).to(device)
    sin_value_x = torch.sin(theta)
    cos_value_x = torch.cos(theta)
    rot_mat_x = torch.tensor([[1, 0, 0, 0],
                    [0, cos_value_x, -sin_value_x, 0],
                    [0, sin_value_x, cos_value_x, 0],
                    [0, 0, 0, 1]]).unsqueeze(0).repeat(c2ws.shape[0], 1, 1).to(device)
    
    phi = torch.deg2rad(torch.tensor(phi)).to(device)
    sin_value_y = torch.sin(phi)
    cos_value_y = torch.cos(phi)
    rot_mat_y = torch.tensor([[cos_value_y, 0, sin_value_y, 0],
                    [0, 1, 0, 0],
                    [-sin_value_y, 0, cos_value_y, 0],
                    [0, 0, 0, 1]]).unsqueeze(0).repeat(c2ws.shape[0], 1, 1).to(device)
    
    c2ws = torch.matmul(rot_mat_x, c2ws)
    c2ws = torch.matmul(rot_mat_y, c2ws)

    return c2ws

