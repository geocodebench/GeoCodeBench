"""
Reference Implementation for compute_angular_velocity_from_coeffs()
This serves as the ground truth for testing LLM-generated implementations.
"""

import jax
import jax.numpy as jnp


@jax.jit
def map_to_lie_algebra(v: jnp.ndarray) -> jnp.ndarray:
    """Hat operator (R^3 -> so(3)).

    Args:
        v: Array of shape (..., 3) representing rotation vectors.

    Returns:
        Skew-symmetric matrices of shape (..., 3, 3).
    """
    vx, vy, vz = v[..., 0], v[..., 1], v[..., 2]
    zeros = jnp.zeros_like(vx)

    row0 = jnp.stack([zeros, -vz, vy], axis=-1)
    row1 = jnp.stack([vz, zeros, -vx], axis=-1)
    row2 = jnp.stack([-vy, vx, zeros], axis=-1)

    return jnp.stack([row0, row1, row2], axis=-2)


@jax.jit
def compute_angular_velocity_from_coeffs(phi: jnp.ndarray, phi_dot: jnp.ndarray) -> jnp.ndarray:
    """Compute angular velocity from rotation vector and its derivative.
    
    Uses the exponential map formula for SO(3).
    
    Args:
        phi: Rotation vector, shape (..., 3)
        phi_dot: Time derivative of rotation vector, shape (..., 3)
        
    Returns:
        Angular velocity, shape (..., 3)
    """
    eps = 1e-8
    phi_norm = jnp.linalg.norm(phi, axis=-1, keepdims=True)  # (..., 1)
    
    # Standard approach: beta = sin(phi/2)^2 / (phi/2)^2, alpha = sin(phi)/phi
    half_phi = phi_norm / 2.0
    sin_half_phi = jnp.sin(half_phi)
    sin_phi = jnp.sin(phi_norm)
    
    # Handle small angles with series expansion
    beta = jnp.where(
        phi_norm < eps,
        1.0 - (phi_norm**2) / 12.0 + (phi_norm**4) / 240.0,  # Series expansion
        (sin_half_phi / half_phi) ** 2
    )
    
    alpha = jnp.where(
        phi_norm < eps,
        1.0 - (phi_norm**2) / 6.0 + (phi_norm**4) / 120.0,  # Series expansion
        sin_phi / phi_norm
    )
    
    # Create identity matrix
    I = jnp.eye(3)
    I = jnp.broadcast_to(I, phi.shape[:-1] + (3, 3))
    
    # Create skew-symmetric matrix
    phi_hat = map_to_lie_algebra(phi)  # (..., 3, 3)
    
    # Rodrigues formula: res = I + 0.5 * beta * phi_hat + (1/phi^2) * (1-alpha) * phi_hat @ phi_hat
    phi_norm_sq = phi_norm ** 2
    
    # Handle division by phi^2 carefully
    factor = jnp.where(
        phi_norm < eps,
        1.0 / 12.0 - (phi_norm**2) / 240.0,  # Series expansion of (1-alpha)/phi^2
        (1.0 - alpha) / phi_norm_sq
    )
    
    # Construct the Jacobian matrix
    res = I + 0.5 * beta[..., None] * phi_hat + factor[..., None] * (phi_hat @ phi_hat)
    
    # Compute omega = res @ phi_dot
    omega = jnp.einsum('...ij,...j->...i', res, phi_dot)
    return omega
