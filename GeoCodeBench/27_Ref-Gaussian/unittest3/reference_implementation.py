"""
Reference Implementation for BSDF functions
This serves as the ground truth for testing LLM-generated implementations.
"""

import math
import torch

specular_epsilon = 1e-4


def bsdf_fresnel_shlick(f0, f90, cosTheta):
    """Fresnel-Schlick approximation for specular reflectance.
    
    Args:
        f0: Reflectance at normal incidence
        f90: Reflectance at grazing angle
        cosTheta: Cosine of the angle between view direction and half vector
    
    Returns:
        Fresnel term
    """
    _cosTheta = torch.clamp(cosTheta, min=specular_epsilon, max=1.0 - specular_epsilon)
    return f0 + (f90 - f0) * (1.0 - _cosTheta) ** 5.0


def bsdf_ndf_ggx(alphaSqr, cosTheta):
    """GGX Normal Distribution Function (NDF).
    
    Args:
        alphaSqr: Square of the roughness parameter
        cosTheta: Cosine of the angle between normal and half vector
    
    Returns:
        Normal distribution term
    """
    _cosTheta = torch.clamp(cosTheta, min=specular_epsilon, max=1.0 - specular_epsilon)
    d = (_cosTheta * alphaSqr - _cosTheta) * _cosTheta + 1
    return alphaSqr / (d * d * math.pi)


def bsdf_lambda_ggx(alphaSqr, cosTheta):
    """GGX Lambda function for Smith masking-shadowing term.
    
    Args:
        alphaSqr: Square of the roughness parameter
        cosTheta: Cosine of the angle
    
    Returns:
        Lambda term
    """
    _cosTheta = torch.clamp(cosTheta, min=specular_epsilon, max=1.0 - specular_epsilon)
    cosThetaSqr = _cosTheta * _cosTheta
    tanThetaSqr = (1.0 - cosThetaSqr) / cosThetaSqr
    res = 0.5 * (torch.sqrt(1 + alphaSqr * tanThetaSqr) - 1.0)
    return res


def bsdf_masking_smith_ggx_correlated(alphaSqr, cosThetaI, cosThetaO):
    """Smith masking-shadowing term for GGX (correlated).
    
    Args:
        alphaSqr: Square of the roughness parameter
        cosThetaI: Cosine of the incident angle
        cosThetaO: Cosine of the outgoing angle
    
    Returns:
        Masking-shadowing term
    """
    lambdaI = bsdf_lambda_ggx(alphaSqr, cosThetaI)
    lambdaO = bsdf_lambda_ggx(alphaSqr, cosThetaO)
    return 1 / (1 + lambdaI + lambdaO)

