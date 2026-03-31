"""
Reference Implementation for coords_grid, yin_to_3d, yang90_from_3d
This serves as the ground truth for testing LLM-generated implementations.
"""

import torch


def coords_grid(b, h, w, homogeneous=False, device=None):
    """
    Generate coordinate grid for image processing.
    
    Args:
        b (int): Batch size
        h (int): Height
        w (int): Width
        homogeneous (bool): Whether to include homogeneous coordinate
        device: PyTorch device
        
    Returns:
        torch.Tensor: Coordinate grid of shape [B, 2, H, W] or [B, 3, H, W]
    """
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w))  # [H, W]

    stacks = [x, y]

    if homogeneous:
        ones = torch.ones_like(x)  # [H, W]
        stacks.append(ones)

    grid = torch.stack(stacks, dim=0).float()  # [2, H, W] or [3, H, W]

    grid = grid[None].repeat(b, 1, 1, 1)  # [B, 2, H, W] or [B, 3, H, W]

    if device is not None:
        grid = grid.to(device)

    return grid


def yin_to_3d(grid, h, w):
    """
    Convert 2D grid coordinates to 3D world coordinates (Yin projection).
    
    Args:
        grid (torch.Tensor): 2D coordinates, shape [N, 2]
        h (int): Height
        w (int): Width
        
    Returns:
        torch.Tensor: 3D world coordinates, shape [N, 3]
    """
    grid_x, grid_y = grid[:, 0], grid[:, 1]

    lat = - grid_y * (torch.pi/2) / (h-1) + torch.pi / 4
    lon = grid_x * (3*torch.pi/2) / (w-1) - 3*torch.pi / 4
    
    x = torch.cos(lat) * torch.sin(lon)
    y = - torch.sin(lat)
    z = torch.cos(lat) * torch.cos(lon)
    world_grid = torch.stack([x, y, z], dim=1)

    return world_grid


def yang90_from_3d(points, h, w):
    """
    Project 3D points to 2D image coordinates (Yang90 projection).
    
    Args:
        points (torch.Tensor): 3D points, shape [N, 3]
        h (int): Height
        w (int): Width
        
    Returns:
        torch.Tensor: Homogeneous 2D coordinates, shape [N, 3]
    """
    points_x, points_y, points_z = points[:, 0], points[:, 1], points[:, 2]
    points_yz = torch.sqrt(points_y**2 + points_z**2)
    
    lat = torch.atan2(-points_x, points_yz)
    lon = torch.atan2(points_y, -points_z)
    
    u = lon * 2 * (w-1) / 3 / torch.pi + (w-1) / 2
    v = - lat * 2 * (h-1) / torch.pi + (h-1) / 2
    ones = torch.ones_like(u)
    grid = torch.stack([u, v, ones], dim=1)
    
    return grid

