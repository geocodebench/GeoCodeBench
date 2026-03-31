"""
Reference Implementation for calc_ARAP_global_solve()
This serves as the ground truth for testing LLM-generated implementations.
"""

import sys
import os
from pathlib import Path

# Add parent directory to path to import from original file (absolute for any cwd)
parent_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(parent_dir))

# Import from the original deformations_dARAP.py file
from deformations_dARAP import (
    calc_ARAP_global_solve,
    Meshes,
    SparseLaplaciansSolvers,
    ARAPEnergyTypeName,
    PostprocessAfterSolveName,
)

# Export the function for testing
__all__ = ['calc_ARAP_global_solve']
