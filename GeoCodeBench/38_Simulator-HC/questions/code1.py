import torch

def getRt_batched(x):
    # x shape: (batch_size, 7), where 4 for quaternion and 3 for translation

    # normalize quaternion (should have been normalized)
    q = x[:, :4]
    q_normalized = q / torch.norm(q, dim=1, keepdim=True)

    # get Rotation from quaternion
    R = torch.zeros(*q_normalized.shape[:-1], 3, 3, device=x.device)
    R[:, 0, 0] = q_normalized[:, 0]**2 + q_normalized[:, 1]**2 - q_normalized[:, 2]**2 - q_normalized[:, 3]**2
    R[:, 0, 1] = 2.0 * (q_normalized[:, 1] * q_normalized[:, 2] - q_normalized[:, 0] * q_normalized[:, 3])
    R[:, 0, 2] = 2.0 * (q_normalized[:, 1] * q_normalized[:, 3] + q_normalized[:, 0] * q_normalized[:, 2])
    R[:, 1, 0] = 2.0 * (q_normalized[:, 1] * q_normalized[:, 2] + q_normalized[:, 0] * q_normalized[:, 3])
    R[:, 1, 1] = q_normalized[:, 0]**2 - q_normalized[:, 1]**2 + q_normalized[:, 2]**2 - q_normalized[:, 3]**2
    R[:, 1, 2] = 2.0 * (q_normalized[:, 2] * q_normalized[:, 3] - q_normalized[:, 0] * q_normalized[:, 1])
    R[:, 2, 0] = 2.0 * (q_normalized[:, 1] * q_normalized[:, 3] - q_normalized[:, 0] * q_normalized[:, 2])
    R[:, 2, 1] = 2.0 * (q_normalized[:, 2] * q_normalized[:, 3] + q_normalized[:, 0] * q_normalized[:, 1])
    R[:, 2, 2] = q_normalized[:, 0]**2 - q_normalized[:, 1]**2 - q_normalized[:, 2]**2 + q_normalized[:, 3]**2

    # get the translation
    t = x[:, -3:]

    return R, t

def reproj2D_batched_noncentral(pts3D, x, cameraoffsets):
    
    # Get R and t for the entire batch
    R, t = getRt_batched(x)

    # Compute pts2D_reproj for the entire batch
    pts2D_reproj = torch.bmm(pts3D, R.transpose(1, 2)) + t.unsqueeze(1)

    pts2D_reproj = pts2D_reproj-cameraoffsets
    norms = torch.norm(pts2D_reproj, dim=2, keepdim=True)
    pts2D_unit = pts2D_reproj / norms

    return pts2D_unit



def upnp_fill_s_batched(quaternions):
    # # Ensure the input is a PyTorch tensor
    # quaternions = torch.tensor(quaternions, dtype=torch.float64)

    # Initialize the output tensor
    s = torch.zeros(quaternions.shape[0], 10, device=quaternions.device)

    # Compute the elements of 's' for each quaternion in the batch
    s[:, 0] = quaternions[:, 0] * quaternions[:, 0]
    s[:, 1] = quaternions[:, 1] * quaternions[:, 1]
    s[:, 2] = quaternions[:, 2] * quaternions[:, 2]
    s[:, 3] = quaternions[:, 3] * quaternions[:, 3]
    s[:, 4] = quaternions[:, 0] * quaternions[:, 1]
    s[:, 5] = quaternions[:, 0] * quaternions[:, 2]
    s[:, 6] = quaternions[:, 0] * quaternions[:, 3]
    s[:, 7] = quaternions[:, 1] * quaternions[:, 2]
    s[:, 8] = quaternions[:, 1] * quaternions[:, 3]
    s[:, 9] = quaternions[:, 2] * quaternions[:, 3]

    return s

def phiMatrix_batch(x):
    """
    Takes a batch of 3D vectors and returns a batch of 3x10 matrices.

    Parameters:
    x: A PyTorch tensor of shape [batch_size, 3, 1]

    Returns:
    A PyTorch tensor of shape [batch_size, 3, 10]
    """
    x1 = x[:, :, 0].unsqueeze(-1)
    x2 = x[:, :, 1].unsqueeze(-1)
    x3 = x[:, :, 2].unsqueeze(-1)

    # Create the 3x10 matrix for each vector in the batch
    zeros = torch.zeros_like(x1)
    twos = 2 * torch.ones_like(x1)

    Phi = torch.stack([
        torch.cat([x1,  x1, -x1, -x1, zeros,  twos * x3, -twos * x2, twos * x2, twos * x3, zeros], dim=2),
        torch.cat([x2, -x2,  x2, -x2, -twos * x3, zeros,  twos * x1, twos * x1, zeros, twos * x3], dim=2),
        torch.cat([x3, -x3, -x3,  x3,  twos * x2, -twos * x1, zeros, zeros, twos * x1, twos * x2], dim=2)
    ], dim=2)

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

    ****EMPTY****

    return M

def UnifiedPnPCoeffall_IJ(f_batch, p_batch, v_batch):
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

    return M, I, J