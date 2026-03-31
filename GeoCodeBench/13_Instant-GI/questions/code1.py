
import torch
from timm.layers import trunc_normal_
from torch import nn
from torch.nn import functional as F

from gaussianimage_rs import inv_sigmoid
from generalizable_model.convnext_unet import ConvNeXtUnet
from generalizable_model.ellipse_process import EllipseProcess
from gsplat import project_gaussians_2d_scale_rot, rasterize_gaussians_sum


def render(xy, scaling, rotation, color, H, W):
    tile_bounds = (
        (W + 16 - 1) // 16,
        (H + 16 - 1) // 16,
        1,
    )
    xys, depths, radii, conics, num_tiles_hit = project_gaussians_2d_scale_rot(
        xy, scaling, rotation, H, W, tile_bounds
    )
    opacity = torch.ones_like(rotation)
    out_img = rasterize_gaussians_sum(
        xys, depths, radii, conics, num_tiles_hit,
        color, opacity, H, W, 16, 16
    )
    # out_img = torch.clamp(out_img, 0, 1)  # [H, W, 3]
    out_img = out_img.view(-1, H, W, 3).permute(0, 3, 1, 2).contiguous()
    return {"render": out_img}



def normalize_tri(triangles):
    # normalize tri
    x1, y1 = triangles[:, 0, 0], triangles[:, 0, 1]
    x2, y2 = triangles[:, 1, 0], triangles[:, 1, 1]
    x3, y3 = triangles[:, 2, 0], triangles[:, 2, 1]
    areas = 0.5 * torch.abs(x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))
    scales = torch.sqrt(1.0 / areas).unsqueeze(-1).unsqueeze(-1)
    tri_centers = triangles.mean(dim=1, keepdim=True)  # [N, 1, 2]
    normalized_tri = (triangles - tri_centers) * scales  # [N, 3, 2]

    return normalized_tri.view(-1, 6)


def sample_operation(triangles, cir_centers, feature_map, image):
    ****EMPTY****
    return map_feature, color_feature


def scaling_activation(x):
    x = 50 * torch.tanh(0.02 * x)
    return torch.where(
        x < 0,
        0.5 * torch.exp(0.5 * x),
        F.softplus(0.25 * x - 2.5, beta=-2, threshold=4) + 3
    )


class InitNet(nn.Module):
    def __init__(self, kernel_size=3):
        super().__init__()

        self.kernel_size = kernel_size
        self.feature_dim = 64
        self.feature_net = ConvNeXtUnet(
            out_channels=self.feature_dim, encoder_name='convnext_base',
            pretrained=True, in_22k=False, in_channels=3, bilinear=False
        )

        self.position_field = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.ReLU(),
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.ReLU(),
            nn.Linear(self.feature_dim, 1),
            nn.Sigmoid(),
        )

        self.feature_reduction = nn.Sequential(
            nn.Linear(self.feature_dim * 4, self.feature_dim * 4),
            nn.ReLU(),
            nn.Linear(self.feature_dim * 4, self.feature_dim * 2),
            nn.ReLU(),
            nn.Linear(self.feature_dim * 2, self.feature_dim),
            nn.LayerNorm(self.feature_dim)
        )

        self.mlp_dim = 4 + 6 + 12 + self.feature_dim  # = 22 + 64 = 86

        self.bc_field = nn.Sequential(
            nn.Linear(self.mlp_dim, self.mlp_dim),
            nn.ReLU(),
            nn.Linear(self.mlp_dim, self.mlp_dim),
            nn.ReLU(),
            nn.Linear(self.mlp_dim, 3),
            nn.Softmax(dim=1)
        )  # [N, 3]

        self.mlp_dim = 8 + 4 + 6 + 12 + self.feature_dim  # = 30 + 64 = 94
        self.scale_rot_field = nn.Sequential(
            nn.Linear(self.mlp_dim, self.mlp_dim),
            nn.ReLU(),
            nn.Linear(self.mlp_dim, self.mlp_dim),
            nn.ReLU(),
            nn.Linear(self.mlp_dim, 3)
        )  # [N, 3]

        self.color_field = nn.Sequential(
            nn.Linear(self.mlp_dim, self.mlp_dim),
            nn.ReLU(),
            nn.Linear(self.mlp_dim, self.mlp_dim),
            nn.ReLU(),
            nn.Linear(self.mlp_dim, 1),
            nn.Sigmoid()
        )  # [N, 3]

        self.apply(self._init_weights)
        self.ell_process = EllipseProcess()

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, image, get_gaussians=False, only_get_pf=False):  # [N, 2], range is [-1, 1]
        feature_map = self.feature_net(image)  # [1, C, H, W]
        B, C, H, W = feature_map.shape
        out_position = self.position_field(feature_map.view(B, C, -1).permute(0, 2, 1))  # [B, H * W, 1]
        out_position = out_position.view(B, H, W)  # [B, H, W]
        if only_get_pf:
            return out_position, image

        triangles, e_center, e_size, e_angle, neighbors = self.ell_process.process(out_position, self.kernel_size)
        # [N, 3, 2], [N, 2], [N, 2], [N, 1]

        map_feature, color_feature = sample_operation(triangles, e_center.unsqueeze(1), feature_map, image)

        reduced_feature = self.feature_reduction(map_feature.view(-1, 4 * self.feature_dim))
        color_feature = color_feature.view(-1, 12)  # [N, 12]
        tri_feature = normalize_tri(triangles)  # [N, 6]
        # tri_feature = triangles.view(-1, 6)  # [N, 6]
        ell_feature = torch.cat((e_center, (e_size[:, 0:1] / (e_size[:, 1:2] + 0.0001)), e_angle.unsqueeze(-1)), dim=1)  # [N, 4]

        mlp_feature = torch.cat((ell_feature, tri_feature, color_feature, reduced_feature), dim=1)

        bc = self.bc_field(mlp_feature).unsqueeze(1)  # [N, 1, 3]
        xy = torch.matmul(bc, triangles).squeeze(1)  # [N, 2]
        xy = torch.clamp(xy, -1 + 1e-6, 1 - 1e-6)
        # The operation of concatenating mlp_feature with neighbors_xy and feeding it into subsequent networks has almost no effect, just ignore its impact
        neighbors_xy = torch.cat(
            [xy, xy[neighbors[:, 0]], xy[neighbors[:, 1]], xy[neighbors[:, 2]]], dim=1
        ).detach()  # [N, 8]
        mlp_feature = torch.cat([neighbors_xy, mlp_feature], dim=1)  # [N, 8 + 4 + 6 + 12 + 64] = [N, 94]

        scale_rot = self.scale_rot_field(mlp_feature)  # [N, 3]
        scaling = scaling_activation(scale_rot[:, :2]) * e_size  # [N, 2]
        inv_e_angle = inv_sigmoid(e_angle.unsqueeze(-1))
        rotation = torch.tanh(scale_rot[:, 2:3]) + inv_e_angle  # [N, 1]
        rotation = torch.sigmoid(rotation)  # [N, 1]
        sampled_color = F.grid_sample(
            image,
            xy.unsqueeze(0).unsqueeze(0).detach(),  # [1, 1, N, 2]
            mode="bilinear",
            align_corners=False
        )  # [1, 3, 1, N]
        sampled_color = sampled_color.squeeze(0).squeeze(1).permute(1, 0)  # [N, 3]
        color = self.color_field(mlp_feature) * sampled_color

        if get_gaussians:
            return xy, scaling, rotation, color, triangles
        else:
            rotation = rotation * 2 * torch.pi
            render_img = render(xy, scaling, rotation, color, H, W)["render"]

            # import matplotlib.pyplot as plt
            # plt.imshow(render_img[0].detach().cpu().numpy().transpose(1, 2, 0))
            # plt.show()
            # exit(0)

            return out_position, render_img, scaling
