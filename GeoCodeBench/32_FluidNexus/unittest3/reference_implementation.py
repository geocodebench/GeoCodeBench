"""
Reference Implementation for update_quaternion()
This serves as the ground truth for testing LLM-generated implementations.
"""

import torch


def update_quaternion(q, omega, delta_t):
    """
    Update quaternion based on angular velocity.
    
    Args:
        q: Quaternion tensor of shape [N, 4] (w, x, y, z format)
        omega: Angular velocity tensor of shape [N, 3] (in body frame)
        delta_t: Time step (scalar or tensor)
    
    Returns:
        q_prime: Updated quaternion tensor of shape [N, 4]
    """
    magnitude_omega = torch.norm(omega, dim=1, keepdim=True)
    half_angle = magnitude_omega * delta_t / 2.0
    delta_q_cos = torch.cos(half_angle)
    delta_q_sin = (
        torch.sin(half_angle) * omega / (magnitude_omega + torch.tensor([1e-8], dtype=torch.float))
    )

    delta_q = torch.cat((delta_q_cos, delta_q_sin), dim=1)

    # Quaternion multiplication
    q0_delta_q0 = q[:, 0:1] * delta_q[:, 0:1]
    cross_product = torch.cross(q[:, 1:], delta_q[:, 1:], dim=1)
    dot_product = (q[:, 1:] * delta_q[:, 1:]).sum(dim=1, keepdim=True)
    q_prime = torch.cat(
        (q0_delta_q0 - dot_product, q[:, 0:1] * delta_q[:, 1:] + delta_q[:, 0:1] * q[:, 1:] + cross_product), dim=1
    )

    return q_prime
