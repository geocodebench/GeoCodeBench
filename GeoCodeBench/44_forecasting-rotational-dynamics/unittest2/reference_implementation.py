"""
Reference Implementation for ddexp_so3()
This serves as the ground truth for testing LLM-generated implementations.
"""

import jax
import jax.numpy as jnp


@jax.jit
def map_to_lie_algebra(v: jnp.ndarray) -> jnp.ndarray:
    r"""Hat operator (\(\mathbb R^3 \to \mathfrak{so}(3)\)).

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
def ddexp_so3(x: jnp.ndarray, z: jnp.ndarray, eps: float = 1e-8) -> jnp.ndarray:
    """Derivative of the exponential map on SO(3).
    
    This computes the derivative of the exponential map dexp_x(z), which is needed
    for second-order dynamics using the derivative of the exponential map.
    
    Args:
        x: Rotation vector of shape (..., 3)
        z: Direction vector of shape (..., 3)
        eps: Numerical threshold for small angles
        
    Returns:
        Derivative matrix of shape (..., 3, 3)
    """
    hatx = map_to_lie_algebra(x)  # (..., 3, 3)
    hatz = map_to_lie_algebra(z)  # (..., 3, 3)
    
    phi = jnp.linalg.norm(x, axis=-1, keepdims=True)[..., None]  # (..., 1, 1)
    
    # Handle small angles with series expansions
    half_phi = phi / 2.0
    sin_half_phi = jnp.sin(half_phi)
    sin_phi = jnp.sin(phi)
    
    # beta = sin(phi/2)^2 / (phi/2)^2
    beta = jnp.where(
        phi < eps,
        1.0 - (phi**2) / 12.0 + (phi**4) / 240.0,  # Series expansion
        (sin_half_phi / half_phi) ** 2
    )
    
    # alpha = sin(phi) / phi
    alpha = jnp.where(
        phi < eps,
        1.0 - (phi**2) / 6.0 + (phi**4) / 120.0,  # Series expansion
        sin_phi / phi
    )
    
    # Compute dot product x·z
    x_dot_z = jnp.sum(x * z, axis=-1, keepdims=True)[..., None]  # (..., 1, 1)
    
    # Terms in the derivative formula
    term1 = 0.5 * beta * hatz
    
    # Handle division by phi^2 carefully
    phi_sq = phi ** 2
    coeff2 = jnp.where(
        phi < eps,
        1.0 / 6.0 - (phi**2) / 120.0,  # Series expansion of (1-alpha)/phi^2
        (1.0 - alpha) / phi_sq
    )
    term2 = coeff2 * (hatx @ hatz + hatz @ hatx)
    
    coeff3 = jnp.where(
        phi < eps,
        -1.0 / 12.0 + (phi**2) / 240.0,  # Series expansion of (alpha-beta)/phi^2
        (alpha - beta) / phi_sq
    )
    term3 = coeff3 * x_dot_z * hatx
    
    coeff4 = jnp.where(
        phi < eps,
        -1.0 / 24.0 + (phi**2) / 720.0,  # Series expansion
        (beta / 2.0 - 3.0 / phi_sq * (1.0 - alpha)) / phi_sq
    )
    term4 = coeff4 * x_dot_z * (hatx @ hatx)
    
    return term1 + term2 + term3 + term4
