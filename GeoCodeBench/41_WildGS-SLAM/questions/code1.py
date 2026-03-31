# Copyright 2024 The GlORIE-SLAM Authors.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn.functional as F
import src.geom.projective_ops as pops

# class CholeskySolver(torch.autograd.Function):
class CholeskySolver():
    @staticmethod

    def apply(H,b):
        ****EMPTY****

        return xs

    def __call__(ctx, H, b):
        # don't crash training if cholesky decomp fails
        ****EMPTY****

        return xs

    @staticmethod
    def backward(ctx, grad_x):
        if ctx.failed:
            return None, None

        ****EMPTY****

        return dH, dz

def block_solve(H, b, ep=0.1, lm=0.0001):
    """ solve normal equations """
    ****EMPTY****
    return x.reshape(B, N, D)


def schur_solve(H, E, C, v, w, ep=0.1, lm=0.0001, sless=False):
    """ solve using shur complement """
    
    ****EMPTY****

    return dx, dz