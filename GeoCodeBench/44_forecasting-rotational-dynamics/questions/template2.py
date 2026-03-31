
"""
Template for LLM Implementation
Copy this file and fill in the function body with LLM-generated code.
Only input and output are provided, no hints.
"""

import jax
import jax.numpy as jnp


@jax.jit
def map_to_lie_algebra(v: jnp.ndarray) -> jnp.ndarray:
    """Hat operator (\(\mathbb R^3 \to \mathfrak{so}(3)\)).

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
    
    # ****EMPTY**** - Fill in the implementation here
    raise NotImplementedError("****EMPTY**** - Please implement the function body")
    
    return term1 + term2 + term3 + term4
