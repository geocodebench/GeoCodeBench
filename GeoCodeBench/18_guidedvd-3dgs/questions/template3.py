
"""
Template for LLM Implementation
Copy this file and fill in the EMPTY sections with LLM-generated code.
"""

import torch
import numpy as np
import copy


def sphere2pose(c2ws_input, theta, phi, r, device, x=None, y=None):
    """
    Helper function to transform poses based on spherical coordinates.
    
    Args:
        c2ws_input: Camera-to-world transformation matrices
        theta: Rotation angle around x-axis (in degrees)
        phi: Rotation angle around y-axis (in degrees)
        r: Translation along z-axis
        device: Device to use (cpu or cuda)
        x: Optional translation along x-axis
        y: Optional translation along y-axis
    
    Returns:
        Transformed camera-to-world matrices
    """
    c2ws = copy.deepcopy(c2ws_input)

    # First translate along world z-axis, then rotate
    c2ws[:, 2, 3] += r
    if x is not None:
        c2ws[:, 1, 3] += y
    if y is not None:
        c2ws[:, 0, 3] += x

    theta = torch.deg2rad(torch.tensor(theta)).to(device)
    sin_value_x = torch.sin(theta)
    cos_value_x = torch.cos(theta)
    rot_mat_x = torch.tensor([[1, 0, 0, 0],
                    [0, cos_value_x, -sin_value_x, 0],
                    [0, sin_value_x, cos_value_x, 0],
                    [0, 0, 0, 1]]).unsqueeze(0).repeat(c2ws.shape[0], 1, 1).to(device)
    
    phi = torch.deg2rad(torch.tensor(phi)).to(device)
    sin_value_y = torch.sin(phi)
    cos_value_y = torch.cos(phi)
    rot_mat_y = torch.tensor([[cos_value_y, 0, sin_value_y, 0],
                    [0, 1, 0, 0],
                    [-sin_value_y, 0, cos_value_y, 0],
                    [0, 0, 0, 1]]).unsqueeze(0).repeat(c2ws.shape[0], 1, 1).to(device)
    
    c2ws = torch.matmul(rot_mat_x, c2ws)
    c2ws = torch.matmul(rot_mat_y, c2ws)

    return c2ws


def get_candidate_poses(self, d_phi, d_theta, fovx, fovy,  
                        which_train_view=5, pc_render_single_view=True, ignore_0_0=False):
    """
    Get candidate camera poses around a reference view.
    
    Args:
        self: Object with necessary attributes (c2ws, device, etc.)
        d_phi: List of phi (horizontal) angle offsets in degrees
        d_theta: List of theta (vertical) angle offsets in degrees
        fovx: Field of view in x direction
        fovy: Field of view in y direction
        which_train_view: Index of the training view to use as reference
        pc_render_single_view: Whether to use single view for point cloud
        ignore_0_0: Whether to ignore the case where both d_phi and d_theta are 0
    
    Returns:
        c2w_candidates: Tensor of candidate camera-to-world transformation matrices
        info_dict: Dictionary containing intermediate results
    """
    idx = which_train_view

    print("Get the candidate poses around idx: {}. ".format(idx))

    if pc_render_single_view: 
        imgs = np.array(self.scene.imgs)[[idx]]
        pcd = [self.pcd[idx]]
    else: 
        imgs = np.array(self.scene.imgs)
        pcd = torch.stack(self.pcd, 0)
    
    img_ori = (self.d_images[idx]['img_ori'].squeeze(0).permute(1, 2, 0) + 1.) / 2.  # [512,704,3] [0,1]
    c2ws = self.c2ws[[idx]]
    principal_points = self.principal_points[[idx]]
    focals = self.focals[[idx]]
    depth = [self.depth[idx]]
    H, W = self.d_H, self.d_W

    depth_avg = depth[-1][H // 2, W // 2]
    radius = depth_avg * self.vc_opts.center_scale

    c2ws, pcd, transform_back = world_point_to_obj_my(
        poses=c2ws, points=torch.stack(pcd) if pc_render_single_view else pcd, 
        k=-1, r=radius, elevation=self.vc_opts.elevation, device=self.device)
    masks = None

    # fovx_degree = min((fovx / (math.pi/2.))*90, 30*2)
    # fovy_degree = min((fovy / (math.pi/2.))*90, 30*2)


    # d_phi = [-fovx_degree/2., -fovx_degree/4., -fovx_degree/8, 0, fovx_degree/8, fovx_degree/4., fovx_degree/2.]
    # d_theta = [-fovy_degree/2., -fovy_degree/4., -fovy_degree/8., 0, fovy_degree/8., fovy_degree/4., fovy_degree/2.]
    # print("=> fov: ", d_phi)

    d_phis = []
    d_thetas = []
    d_rs = []
    ****EMPTY****

    c2w_candidates = []
    ****EMPTY****
    
    return c2w_candidates, \
        {"c2ws": c2ws, "d_phis": d_phis, "d_thetas": d_thetas, "d_rs": d_rs, "transform_back": transform_back}


# Mock function for testing
def world_point_to_obj_my(poses, points, k, r, elevation, device):
    """Mock function that simulates coordinate transformation."""
    transform_back = torch.eye(4, device=device).unsqueeze(0)
    return poses, points, transform_back
