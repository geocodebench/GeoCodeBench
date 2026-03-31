"""
Reference Implementation for Dict_pose_acc.forward() function
This serves as the ground truth for testing LLM-generated implementations.
"""

import torch.nn as nn
import torch
import torch.nn.functional as F
from utils import embedder


class Dict_pose_acc(nn.Module):
    def __init__(
        self,
        rank,
        half,
        latent_in = [4],
        dropout_prob = 0.0,
        dropout = [0, 1, 2, 3, 4, 5, 6, 7],
        norm_layers = [0, 1, 2, 3, 4, 5, 6, 7],
        positional_enc=False,
        n_positional_freqs=8,
        n_alpha_epochs=0,
    ):
        super(Dict_pose_acc, self).__init__()
        self.U = nn.ModuleList([])
        self.Vt = nn.ModuleList([])
        self.rank = rank
        self.half = half
        self.res_u = nn.ModuleList([])
        self.res_vt = nn.ModuleList([])
        self.bn = nn.ModuleList([])
        self.latent_in = latent_in
        self.dropout_prob = dropout_prob
        self.dropout = dropout
        self.norm_layers = norm_layers
        self.num_layers = 0
        self.relu = nn.ReLU()
        self.th = nn.Tanh()

        if positional_enc:
            self.n_positional_freqs = n_positional_freqs
            self.pos_embedder, pos_embedder_out_dim = embedder.get_embedder_nerf(
                n_positional_freqs, input_dims=3
            )
            self.n_alpha_epochs = n_alpha_epochs
            self.alpha_const = n_positional_freqs / n_alpha_epochs if n_alpha_epochs > 0 else self.n_positional_freqs

    def init_para(self, bias, U, Vt):
        self.num_layers = len(U)

        for i in range(len(U)):
            self.U.append(
                nn.Linear(U[i].shape[1], U[i].shape[0], bias=True)
            )
            self.Vt.append(
                nn.Linear(Vt[i].shape[1], Vt[i].shape[0], bias=False)
            )

            self.U[i].weight = nn.Parameter(U[i])
            self.U[i].bias = nn.Parameter(bias[i])
            self.Vt[i].weight = nn.Parameter(Vt[i])

            if i >= self.half and self.rank:
                rk = self.rank
                self.res_u.append(
                    nn.Linear(rk, U[i].shape[0], bias=False)
                )
                nn.init.normal_(self.res_u[i - self.half].weight, mean=0.0, std=0.001)
                self.res_vt.append(
                   nn.Linear(Vt[i].shape[1], rk, bias=False)
                )
                nn.init.normal_(self.res_vt[i - self.half].weight, mean=0.0, std=0.001)

    def forward(self, input, Sigma, bs):
        xyz = input[:, -3:]

        if hasattr(self, "pos_embedder"):
            input_pos_embed = self.pos_embedder(xyz, self.alpha_const)
            x = torch.cat([input[:, :-3], input_pos_embed], 1)
            input_embed = x.clone()
        else:
            x = input

        for i in range(len(self.U)):
            if i in self.latent_in:
                if hasattr(self, "pos_embedder"):
                    x = torch.cat([x, input_embed], 1)
                else:
                    x = torch.cat([x, input], 1)

            ori_len = self.Vt[i].out_features

            x1 = self.Vt[i](x)
            Sig = torch.exp(Sigma[:, i, :ori_len])
            Sig = Sig.unsqueeze(1)
            sig_repeat = Sig.expand(-1, int(x1.shape[0] / bs), -1)
            sig_repeat = sig_repeat.reshape(-1, sig_repeat.shape[-1])
            x1 = x1 * sig_repeat

            x1 = self.U[i](x1)

            if i >= self.half and self.rank:
                x2 = self.res_vt[i - self.half](x)
                res_sig = torch.exp(Sigma[:, i, ori_len:ori_len+self.rank])
                res_sig = res_sig.unsqueeze(1)
                res_sig_repeat = res_sig.expand(-1, int(x1.shape[0] / bs), -1)
                res_sig_repeat = res_sig_repeat.reshape(-1, res_sig_repeat.shape[-1])
                x2 = x2 * res_sig_repeat
                x2 = self.res_u[i - self.half](x2)
                x = x1 + x2
            else:
                x = x1

            if i < self.num_layers - 1:
                x = self.relu(x)
                if self.dropout is not None and i in self.dropout:
                    x = F.dropout(x, p=self.dropout_prob, training=self.training)

        xyz_warped = xyz + x

        return xyz_warped
