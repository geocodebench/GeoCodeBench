"""
Reference Implementation for UnifiedPnPCoeff()
This serves as the ground truth for testing LLM-generated implementations.
"""

import torch


def phiMatrix_batch(x):
    """
    Takes a batch of 3D vectors and returns a batch of 3x10 matrices.

    Parameters:
    x: A PyTorch tensor of shape [batch_size, nPts, 3, 1] or [batch_size, 3, 1]

    Returns:
    A PyTorch tensor of shape [batch_size, nPts, 3, 10] or [batch_size, 3, 10]
    """
    # Handle both [batch_size, nPts, 3, 1] and [batch_size, 3, 1] cases
    if x.dim() == 4:
        batch_size, nPts, _, _ = x.shape
        x = x.view(batch_size * nPts, 3, 1)
        need_reshape = True
    else:
        need_reshape = False
    
    x1 = x[:, 0, 0].unsqueeze(-1).unsqueeze(-1)
    x2 = x[:, 1, 0].unsqueeze(-1).unsqueeze(-1)
    x3 = x[:, 2, 0].unsqueeze(-1).unsqueeze(-1)

    # Create the 3x10 matrix for each vector in the batch
    zeros = torch.zeros_like(x1)
    twos = 2 * torch.ones_like(x1)

    Phi = torch.stack([
        torch.cat([x1,  x1, -x1, -x1, zeros,  twos * x3, -twos * x2, twos * x2, twos * x3, zeros], dim=2),
        torch.cat([x2, -x2,  x2, -x2, -twos * x3, zeros,  twos * x1, twos * x1, zeros, twos * x3], dim=2),
        torch.cat([x3, -x3, -x3,  x3,  twos * x2, -twos * x1, zeros, zeros, twos * x1, twos * x2], dim=2)
    ], dim=2)

    if need_reshape:
        Phi = Phi.view(batch_size, nPts, 3, 10)
    
    return Phi


def UnifiedPnPCoeff(f_batch, p_batch, v_batch):
    # f image ray (unit norm)
    # p 3D world points

    device = f_batch.device

    nPts = f_batch.size(1)
    nBatch = f_batch.size(0)

    f_batch_T = f_batch.transpose(-2, -1)  # Shape: [batch_size, 1, 3]

    # Compute F without a loop
    F = torch.bmm(f_batch_T, f_batch)

    H_inv = nPts * torch.eye(3, device=device) - F
    H = torch.inverse(H_inv)

    P = (torch.einsum('bmi,bmj->bmij', f_batch, f_batch) - torch.eye(3, device=device).unsqueeze(0).unsqueeze(0)) 

    # Compute I, J, and M without loops
    # v_batch = torch.zeros(f_batch.size(0), nPts, 3, device=device)  # central case for now

    # Compute Phi for all points in all batches at once
    Phi_batch = phiMatrix_batch(p_batch)  # Assuming this function is batch-aware and the output shape is (nBatch, nPts, 3, 10)

    # Compute Vk for all points in all batches
    Vk_batch = torch.matmul(H.unsqueeze(1), P)  # Shape: (nBatch, nPts, 3, 3)

    # Batch-wise computation of I and J
    I = torch.einsum('bnij,bnjk->bik', Vk_batch, Phi_batch)  # Shape: (nBatch, 3, 10)
    J = torch.einsum('bnij,bnj->bi', Vk_batch, v_batch)  # Shape: (nBatch, 3, 1)

    # Prepare Ai and bi for AA, C, gamma
    Ai_batch = torch.einsum('bnij,bnjk->bnik', P, Phi_batch + I.unsqueeze(1))  # Shape: (nBatch, nPts, 3, 10)
    bi_batch = -torch.einsum('bnij,bnj->bni', P, v_batch + J.unsqueeze(1))  # Shape: (nBatch, nPts, 3, 1)

    AA_batch = torch.einsum('bnij,bnjk->bik', Ai_batch.transpose(-2,-1), Ai_batch)
    C_batch = torch.einsum('bnij,bnjk->bik', bi_batch.unsqueeze(2), Ai_batch)
    gamma_batch = torch.einsum('bnij,bnjk->bik', bi_batch.unsqueeze(2), bi_batch.unsqueeze(3))

    M_up = torch.cat((AA_batch, C_batch), dim=1)
    M_low = torch.cat((C_batch.transpose(1,2), gamma_batch), dim=1)

    M = torch.cat((M_up, M_low), dim=2)

    # Uk_batch = torch.einsum('bim,bjmn->bijn', f_batch, Vk_batch) # lack a fi in each 2nd and 3rd dimensions need to be add in get t

    return M
