"""
Reference Implementation for rotmat2quaternion and normal2rotation
This serves as the ground truth for testing LLM-generated implementations.
"""

import torch


def rotmat2quaternion(R, normalize=False):
    """Convert rotation matrix to quaternion.
    
    Args:
        R: Rotation matrix of shape (N, 3, 3)
        normalize: Whether to normalize the quaternion
    
    Returns:
        q: Quaternion of shape (N, 4) in format [w, x, y, z]
    """
    tr = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2] + 1e-6
    r = torch.sqrt(1 + tr) / 2
    
    q = torch.stack([
        r,
        (R[:, 2, 1] - R[:, 1, 2]) / (4 * r),
        (R[:, 0, 2] - R[:, 2, 0]) / (4 * r),
        (R[:, 1, 0] - R[:, 0, 1]) / (4 * r)
    ], -1)
    
    if normalize:
        q = torch.nn.functional.normalize(q, dim=-1)

    return q


def normal2rotation(n):
    """Construct a rotation from normal vector.
    
    Args:
        n: Normal vector of shape (N, 3)
    
    Returns:
        q: Quaternion of shape (N, 4) in format [w, x, y, z]
    """
    # construct a random rotation matrix from normal
    # it would better be positive definite and orthogonal
    n = torch.nn.functional.normalize(n)
    # w0 = torch.rand_like(n)
    w0 = torch.tensor([[1, 0, 0]]).expand(n.shape).to(n.device)
    R0 = w0 - torch.sum(w0 * n, -1, True) * n
    R0 *= torch.sign(R0[:, :1])
    R0 = torch.nn.functional.normalize(R0)
    R1 = torch.cross(n, R0, dim=-1)
    
    # i = 7859
    # print(R1[i])
    R1 *= torch.sign(R1[:, 1:2]) * torch.sign(n[:, 2:])
    # print(R1[i])
    R = torch.stack([R0, R1, n], -1)
    # print(R[i], torch.det(R).sum(), torch.trace(R[i]))
    q = rotmat2quaternion(R)
    # print(q[i], torch.norm(q[i]))
    # R = quaternion2rotmat(q)
    # print(R[i])
    # for i in range(len(q)):
    #     if torch.isnan(q[i].sum()):
    #         print(i)
    # exit()
    return q

