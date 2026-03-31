"""
Reference Implementation for init_from_coeff
This serves as the ground truth for testing LLM-generated implementations.
"""

import torch
import os
import json


class MockDataset:
    """Mock dataset class for testing."""
    def __init__(self, source_path):
        self.source_path = source_path


def init_from_coeff(coeff, dataset, ref_points):
    """
    Apply camera distortion coefficients to reference points.
    
    Args:
        coeff: Distortion coefficients (list of length 2, 3, 4, or 8)
        dataset: Dataset object with source_path attribute
        ref_points: Reference points tensor, shape [..., 2]
    
    Returns:
        Distorted reference points with same shape as input
    """
    r = torch.sqrt(torch.sum(ref_points**2, dim=-1, keepdim=True))
    inv_r = 1 / r
    theta = torch.atan(r)
    
    # Note: In test environment, the following file reading code is not executed
    # because test uses MockDataset with non-existent paths.
    # The coeff parameter is directly provided by the test cases.
    if os.path.exists(os.path.join(dataset.source_path, 'fish/sparse/0/cameras.bin')):
        # This branch won't execute in tests (file doesn't exist)
        # read_intrinsics_binary would be imported from scene.dataset_readers in real usage
        pass
    elif os.path.exists(os.path.join(dataset.source_path, 'cameras.json')):
        # This branch won't execute in tests (file doesn't exist)
        pass
    
    if len(coeff) == 4:
        ref_points = ref_points * (inv_r * (theta + coeff[0] * theta**3 + coeff[1] * theta**5 + coeff[2] * theta**7 + coeff[3] * theta**9))
    elif len(coeff) == 2:
        ref_points = ref_points * (1 + coeff[0] * r**2 + coeff[1] * r**4)
    elif len(coeff) == 3:
        ref_points = ref_points * (1 + coeff[0] * r**2 + coeff[1] * r**4 + coeff[2] * r**6)
    elif len(coeff) == 8:
        x_n, y_n = ref_points[..., 0], ref_points[..., 1]
        p1, p2 = coeff[5], coeff[6]
        r_squared = x_n**2 + y_n**2
        tangential_distortion = torch.stack([
            2 * p1 * x_n * y_n + p2 * (r_squared + 2 * x_n**2),
            p1 * (r_squared + 2 * y_n**2) + 2 * p2 * x_n * y_n
        ], dim=-1)
        #ref_points = ref_points * (inv_r * (theta + coeff[0] * theta**3 + coeff[1] * theta**5 + coeff[2] * theta**7)) + tangential_distortion
        ref_points = ref_points * (inv_r * (theta + coeff[0] * theta**3 + coeff[1] * theta**5 + coeff[2] * theta**7))
    else:
        ref_points = ref_points
    
    print(f"using coeff: {coeff}")
    
    return ref_points

