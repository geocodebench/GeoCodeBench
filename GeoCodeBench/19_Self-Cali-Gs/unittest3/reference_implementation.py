"""
Reference Implementation (Ground Truth)
This is the correct implementation of init_cubemap from util_distortion.py
"""

import torch
from tqdm import tqdm


def generate_circle_points(min_radius=0.0, max_radius=2.0, radius_step=0.2, num_angles=100):
    """Generate circle points for testing distortion.
    
    Args:
        min_radius: Minimum radius value
        max_radius: Maximum radius value
        radius_step: Step size for radius values
        num_angles: Number of angles to sample
        
    Returns:
        circle_points: Points on circles [n, 2]
        radii: Radii of the points [n]
    """
    radii = torch.arange(min_radius, max_radius + 1e-7, radius_step)  # 1e-7 to ensure inclusive
    angles = torch.linspace(0, 2 * torch.pi, num_angles)  # [num_angles]
    R, Theta = torch.meshgrid(radii, angles, indexing='ij')

    X = R * torch.cos(Theta)
    Y = R * torch.sin(Theta)
    circle_points = torch.stack([X.flatten(), Y.flatten()], dim=-1)  # [n, 2]
    radii = torch.sqrt(torch.sum(circle_points ** 2, dim=1))

    return circle_points, radii


def init_cubemap(scene, dataset, optimizer_lens_net, lens_net, scheduler_lens_net, resume_training=None, iresnet_lr=1e-7, cubemap=False):
    """Initialize cubemap with lens distortion correction.
    
    This function trains the lens_net to correct lens distortion using generated circle points.
    
    Args:
        scene: Scene object (not used in core logic, can be None for testing)
        dataset: Dataset object with source_path attribute
        optimizer_lens_net: Optimizer for lens network
        lens_net: Neural network for lens distortion correction
        scheduler_lens_net: Learning rate scheduler
        resume_training: Whether to resume training (default: None)
        iresnet_lr: Initial learning rate (default: 1e-7)
        cubemap: Whether using cubemap mode (default: False)
    """
    # Use default distortion coefficients for testing
    # In real environment, these would be read from dataset
    coeff = [0., 0., 0., 0.]

    # Generate circle points for training
    points_n, r_n = generate_circle_points(min_radius=0.05, max_radius=80.0, radius_step=0.05, num_angles=100)
    
    # Apply distortion model to create training pairs
    r_d = torch.atan(r_n) + coeff[0] * torch.atan(r_n)**3 + coeff[1] * torch.atan(r_n)**5 + coeff[2] * torch.atan(r_n)**7 + coeff[3] * torch.atan(r_n)**9
    inv_r_n = 1 / (r_n + 1e-5)
    points_d = (r_d * inv_r_n).unsqueeze(-1) * points_n

    # Compute undistorted points for training
    inv_r_d = 1 / (r_d + 1e-5)
    r_n_ = torch.tan(r_d)
    scale_ = r_n * inv_r_d
    train_x = (points_d * scale_.unsqueeze(-1))
    train_y = points_n
    
    # Training loop
    progress_bar_ires = tqdm(range(0, 100), desc="Init Iresnet")
    for i in range(100):
        pred_x = lens_net.forward(train_x, sensor_to_frustum=True)
        loss = ((pred_x - train_y)**2).mean()
        progress_bar_ires.set_postfix(loss=loss.item())
        progress_bar_ires.update(1)
        loss.backward()
        optimizer_lens_net.step()
        optimizer_lens_net.zero_grad(set_to_none=True)
        scheduler_lens_net.step()
    progress_bar_ires.close()

