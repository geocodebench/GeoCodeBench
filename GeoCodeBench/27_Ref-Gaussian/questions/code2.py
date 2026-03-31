import torch
import numpy as np
import nvdiffrast.torch as dr
from .general_utils import safe_normalize, flip_align_view
from utils.sh_utils import eval_sh
import kornia

env_rayd1 = None
FG_LUT = torch.from_numpy(np.fromfile("assets/bsdf_256_256.bin", dtype=np.float32).reshape(1, 256, 256, 2)).cuda()
def init_envrayd1(H,W):
    i, j = np.meshgrid(
        np.linspace(-np.pi, np.pi, W, dtype=np.float32),
        np.linspace(0, np.pi, H, dtype=np.float32),
        indexing='xy'
    )
    xy1 = np.stack([i, j], axis=2)
    z = np.cos(xy1[..., 1])
    x = np.sin(xy1[..., 1])*np.cos(xy1[...,0])
    y = np.sin(xy1[..., 1])*np.sin(xy1[...,0])
    global env_rayd1
    env_rayd1 = torch.tensor(np.stack([x,y,z], axis=-1)).cuda()

def get_env_rayd1(H,W):
    if env_rayd1 is None:
        init_envrayd1(H,W)
    return env_rayd1

env_rayd2 = None
def init_envrayd2(H,W):
    gy, gx = torch.meshgrid(torch.linspace( 0.0 + 1.0 / H, 1.0 - 1.0 / H, H, device='cuda'), 
                            torch.linspace(-1.0 + 1.0 / W, 1.0 - 1.0 / W, W, device='cuda'),
                            # indexing='ij')
                            )
    
    sintheta, costheta = torch.sin(gy*np.pi), torch.cos(gy*np.pi)
    sinphi, cosphi     = torch.sin(gx*np.pi), torch.cos(gx*np.pi)
    
    reflvec = torch.stack((
        sintheta*sinphi, 
        costheta, 
        -sintheta*cosphi
        ), dim=-1)
    global env_rayd2
    env_rayd2 = reflvec

def get_env_rayd2(H,W):
    if env_rayd2 is None:
        init_envrayd2(H,W)
    return env_rayd2



pixel_camera = None
def sample_camera_rays(HWK, R, T):
    H,W,K = HWK
    R = R.T # NOTE!!! the R rot matrix is transposed save in 3DGS
    
    global pixel_camera
    if pixel_camera is None or pixel_camera.shape[0] != H:
        K = K.astype(np.float32)
        i, j = np.meshgrid(np.arange(W, dtype=np.float32),
                        np.arange(H, dtype=np.float32),
                        indexing='xy')
        xy1 = np.stack([i, j, np.ones_like(i)], axis=2)
        pixel_camera = np.dot(xy1, np.linalg.inv(K).T)
        pixel_camera = torch.tensor(pixel_camera).cuda()

    rays_o = (-R.T @ T.unsqueeze(-1)).flatten()
    pixel_world = (pixel_camera - T[None, None]).reshape(-1, 3) @ R
    rays_d = pixel_world - rays_o[None]
    rays_d = rays_d / torch.norm(rays_d, dim=1, keepdim=True)
    rays_d = rays_d.reshape(H,W,3)
    return rays_d, rays_o

def sample_camera_rays_unnormalize(HWK, R, T):
    H,W,K = HWK
    R = R.T # NOTE!!! the R rot matrix is transposed save in 3DGS
    
    global pixel_camera
    if pixel_camera is None or pixel_camera.shape[0] != H:
        K = K.astype(np.float32)
        i, j = np.meshgrid(np.arange(W, dtype=np.float32),
                        np.arange(H, dtype=np.float32),
                        indexing='xy')
        xy1 = np.stack([i, j, np.ones_like(i)], axis=2)
        pixel_camera = np.dot(xy1, np.linalg.inv(K).T)
        pixel_camera = torch.tensor(pixel_camera).cuda()

    rays_o = (-R.T @ T.unsqueeze(-1)).flatten()
    pixel_world = (pixel_camera - T[None, None]).reshape(-1, 3) @ R
    rays_d = pixel_world - rays_o[None]
    rays_d = rays_d.reshape(H,W,3)
    return rays_d, rays_o

def reflection(w_o, normal):
    NdotV = torch.sum(w_o*normal, dim=-1, keepdim=True)
    w_k = 2*normal*NdotV - w_o
    return w_k, NdotV





def get_specular_color_surfel(envmap: torch.Tensor, albedo, HWK, R, T, normal_map, render_alpha, scaling_modifier = 1.0, refl_strength = None, roughness = None, pc=None, surf_depth=None, indirect_light=None): #RT W2C
    global FG_LUT
    H,W,K = HWK
    rays_cam, rays_o = sample_camera_rays(HWK, R, T)
    w_o = -rays_cam
    rays_refl, NdotV = reflection(w_o, normal_map)
    rays_refl = safe_normalize(rays_refl)

    # Query BSDF
    fg_uv = torch.cat([NdotV, roughness], -1).clamp(0, 1) 
    fg = dr.texture(FG_LUT, fg_uv.reshape(1, -1, 1, 2).contiguous(), filter_mode="linear", boundary_mode="clamp").reshape(1, H, W, 2) 
    # Compute direct light
    direct_light = envmap(rays_refl, roughness=roughness)
    specular_weight = ((0.04 * (1 - refl_strength) + albedo * refl_strength) * fg[0][..., 0:1] + fg[0][..., 1:2]) 
    
    # visibility
    visibility = torch.ones_like(render_alpha)
    if pc.ray_tracer is not None and indirect_light is not None:
        mask = (render_alpha>0)[..., 0]
        rays_cam, rays_o = sample_camera_rays_unnormalize(HWK, R, T)
        w_o = safe_normalize(-rays_cam)
        # import pdb;pdb.set_trace() 
        rays_refl, _ = reflection(w_o, normal_map)
        rays_refl = safe_normalize(rays_refl)
        intersections = rays_o + surf_depth.permute(1, 2, 0) * rays_cam
        # import pdb;pdb.set_trace()
        _, _, depth = pc.ray_tracer.trace(intersections[mask], rays_refl[mask])
        visibility[mask] = (depth >= 10).float().unsqueeze(-1)
    
        # indirect light
        specular_light = direct_light * visibility + (1 - visibility) * indirect_light
        indirect_color = (1 - visibility) * indirect_light * render_alpha * specular_weight
    else:
        specular_light = direct_light
    
    # Compute specular color
    specular_raw = specular_light * render_alpha
    specular = specular_raw * specular_weight
    

    if indirect_light is not None:
        extra_dict = {
            "visibility": visibility.permute(2,0,1),
            "indirect_light": indirect_light.permute(2,0,1),
            "direct_light": direct_light.permute(2,0,1),
            "indirect_color": indirect_color.permute(2,0,1)
        } 
    else:
        extra_dict = None
        
    return specular.permute(2,0,1), extra_dict






def get_full_color_volume(envmap: torch.Tensor, xyz, albedo, HWK, R, T, normal_map, render_alpha, scaling_modifier = 1.0, refl_strength = None, roughness = None): #RT W2C
    global FG_LUT
    _, rays_o = sample_camera_rays(HWK, R, T)
    N, _ = normal_map.shape
    rays_o = rays_o.expand(N, -1)
    w_o = safe_normalize(rays_o - xyz)
    rays_refl, NdotV = reflection(w_o, normal_map)
    rays_refl = safe_normalize(rays_refl)

    # Query BSDF
    fg_uv = torch.cat([NdotV, roughness], -1).clamp(0, 1) # 计算BSDF参数
    # fg = dr.texture(FG_LUT, fg_uv.reshape(1, -1, 1, 2).contiguous(), filter_mode="linear", boundary_mode="clamp").reshape(1, H, W, 2) 
    fg_uv = fg_uv.unsqueeze(0).unsqueeze(2)  # [1, N, 1, 2]
    fg = dr.texture(FG_LUT, fg_uv, filter_mode="linear", boundary_mode="clamp").squeeze(2).squeeze(0)  # [N, 2]
    # Compute diffuse
    diffuse = envmap(normal_map, mode="diffuse") * (1-refl_strength) * albedo
    # Compute specular
    specular = envmap(rays_refl, roughness=roughness) * ((0.04 * (1 - refl_strength) + albedo * refl_strength) * fg[0][..., 0:1] + fg[0][..., 1:2]) 

    return diffuse, specular




def get_full_color_volume_indirect(envmap: torch.Tensor, xyz, albedo, HWK, R, T, normal_map, render_alpha, scaling_modifier = 1.0, refl_strength = None, roughness = None, pc=None, indirect_light=None): #RT W2C
    global FG_LUT
    _, rays_o = sample_camera_rays(HWK, R, T)
    N, _ = normal_map.shape
    rays_o = rays_o.expand(N, -1)
    w_o = safe_normalize(rays_o - xyz)
    rays_refl, NdotV = reflection(w_o, normal_map)
    rays_refl = safe_normalize(rays_refl)

    # visibility
    visibility = torch.ones_like(render_alpha)
    if pc.ray_tracer is not None:
        mask = (render_alpha>0).squeeze()
        intersections = xyz
        _, _, depth = pc.ray_tracer.trace(intersections[mask], rays_refl[mask])
        visibility[mask] = (depth >= 10).unsqueeze(1).float()

    # Query BSDF
    ****EMPTY****
    extra_dict = {
        "visibility": visibility,
        "direct_light": direct_light,
    }

    return diffuse, specular, extra_dict





# def get_refl_color(envmap: torch.Tensor, HWK, R, T, normal_map): #RT W2C
#     rays_d, _ = sample_camera_rays(HWK, R, T)
#     rays_d, _ = reflection(rays_d, normal_map)
#     return envmap(rays_d, mode="pure_env").permute(2,0,1)