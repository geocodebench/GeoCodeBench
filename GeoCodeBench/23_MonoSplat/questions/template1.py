
"""
Template for LLM Implementation
Copy this file and fill in the function body with LLM-generated code.
"""

from jaxtyping import Float
from torch import Tensor


def relative_disparity_to_depth(
    relative_disparity: Float[Tensor, "*#batch"],
    near: Float[Tensor, "*#batch"],
    far: Float[Tensor, "*#batch"],
    eps: float = 1e-10,
) -> Float[Tensor, " *batch"]:
    """Convert relative disparity, where 0 is near and 1 is far, to depth."""
    # TODO: Fill in LLM-generated code here
    
    raise NotImplementedError("Please implement this function")


def depth_to_relative_disparity(
    depth: Float[Tensor, "*#batch"],
    near: Float[Tensor, "*#batch"],
    far: Float[Tensor, "*#batch"],
    eps: float = 1e-10,
) -> Float[Tensor, " *batch"]:
    """Convert depth to relative disparity, where 0 is near and 1 is far"""
    # TODO: Fill in LLM-generated code here
    
    raise NotImplementedError("Please implement this function")
