"""
Reference Implementation for bsdf_pbr_specular
This serves as the ground truth for testing LLM-generated implementations.
"""

import math
import torch

specular_epsilon = 1e-4

def _dot(x, y):
    return torch.sum(x*y, -1, keepdim=True)

def _safe_normalize(x):
    return torch.nn.functional.normalize(x, dim = -1)

def bsdf_fresnel_shlick(f0, f90, cosTheta):
    _cosTheta = torch.clamp(cosTheta, min=specular_epsilon, max=1.0 - specular_epsilon)
    return f0 + (f90 - f0) * (1.0 - _cosTheta) ** 5.0

def bsdf_ndf_ggx(alphaSqr, cosTheta):
    _cosTheta = torch.clamp(cosTheta, min=specular_epsilon, max=1.0 - specular_epsilon)
    d = (_cosTheta * alphaSqr - _cosTheta) * _cosTheta + 1
    return alphaSqr / (d * d * math.pi)

def bsdf_lambda_ggx(alphaSqr, cosTheta):
    _cosTheta = torch.clamp(cosTheta, min=specular_epsilon, max=1.0 - specular_epsilon)
    cosThetaSqr = _cosTheta * _cosTheta
    tanThetaSqr = (1.0 - cosThetaSqr) / cosThetaSqr
    res = 0.5 * (torch.sqrt(1 + alphaSqr * tanThetaSqr) - 1.0)
    return res

def bsdf_masking_smith_ggx_correlated(alphaSqr, cosThetaI, cosThetaO):
    lambdaI = bsdf_lambda_ggx(alphaSqr, cosThetaI)
    lambdaO = bsdf_lambda_ggx(alphaSqr, cosThetaO)
    return 1 / (1 + lambdaI + lambdaO)

def bsdf_pbr_specular(col, nrm, wo, wi, alpha, min_roughness=0.08):
    """PBR specular BRDF using GGX distribution.
    
    Args:
        col: Color/reflectance (shape: [..., 3])
        nrm: Normal vector (shape: [..., 3])
        wo: View/outgoing direction (shape: [..., 3])
        wi: Light/incoming direction (shape: [..., 3])
        alpha: Roughness parameter (shape: [..., 1])
        min_roughness: Minimum roughness value (default: 0.08)
    
    Returns:
        Specular reflection value (shape: [..., 3])
    """
    _alpha = torch.clamp(alpha, min=min_roughness*min_roughness, max=1.0)
    alphaSqr = _alpha * _alpha

    h = _safe_normalize(wo + wi)
    woDotN = _dot(wo, nrm)
    wiDotN = _dot(wi, nrm)
    woDotH = _dot(wo, h)
    nDotH  = _dot(nrm, h)

    D = bsdf_ndf_ggx(alphaSqr, nDotH)
    G = bsdf_masking_smith_ggx_correlated(alphaSqr, woDotN, wiDotN)
    F = bsdf_fresnel_shlick(col, 1, woDotH)

    w = F * D * G * 0.25 / torch.clamp(woDotN, min=specular_epsilon)

    frontfacing = (woDotN > specular_epsilon) & (wiDotN > specular_epsilon)
    return torch.where(frontfacing, w, torch.zeros_like(w))

