"""
Reference Implementation for Dict_exp.forward() method
This serves as the ground truth for testing LLM-generated implementations.

Description: Forward pass of Dict_exp network with positional encoding and Sigma scaling.
"""

import torch.nn as nn
import torch
import torch.nn.functional as F
from utils import embedder


class Dict_exp(nn.Module):
    def __init__(
        self,
        bias,
        U,
        Sigma,
        Vt,
        latent_in=[4],
        dropout_prob=0.0,
        dropout=None,
        positional_enc=True,
        n_positional_freqs=8,
    ):
        super(Dict_exp, self).__init__()
        self.U_w = U
        self.U = nn.ModuleList([])
        self.Sigma = nn.ParameterList([])  # self.sigma = ln(sigma)
        self.Vt_w = Vt
        self.Vt = nn.ModuleList([])
        self.latent_in = latent_in
        self.dropout_prob = dropout_prob
        self.dropout = dropout
        self.num_layers = len(U)
        self.relu = nn.ReLU()
        self.th = nn.Tanh()

        for i in range(len(U)):
            self.U.append(
                nn.Linear(U[i].shape[1], U[i].shape[0], bias=True)
            )
            self.Sigma.append(
                nn.Parameter(Sigma[i])
            )
            self.Vt.append(
                nn.Linear(Vt[i].shape[1], Vt[i].shape[0], bias=False)
            )

        for i in range(len(self.U)):
            self.U[i].weight = nn.Parameter(U[i])
            self.U[i].bias = nn.Parameter(bias[i])
            self.Vt[i].weight = nn.Parameter(Vt[i])

        if positional_enc:
            self.n_positional_freqs = n_positional_freqs
            self.pos_embedder, pos_embedder_out_dim = embedder.get_embedder_nerf(
                self.n_positional_freqs, input_dims=3, i=0
            )

    def forward(self, input):
        """
        Forward pass of Dict_exp network.
        
        Args:
            input: Input tensor of shape (N, D) where last 3 dimensions are xyz coordinates
        
        Returns:
            Output tensor after passing through the network
        """
        Sigma_m = []
        for i, sigma in enumerate(self.Sigma):
            Sigma_m.append(torch.exp(sigma))

        if hasattr(self, "pos_embedder"):
            xyz = input[:, -3:]
            input_pos_embed = self.pos_embedder(xyz, self.n_positional_freqs)
            x = torch.cat([input[:, :-3], input_pos_embed], 1)
            input_embed = x.clone()
        else:
            x = input

        for i, Sig in enumerate(Sigma_m):
            if i in self.latent_in:
                if hasattr(self, "pos_embedder"):
                    x = torch.cat([x, input_embed], 1)
                else:
                    x = torch.cat([x, input], 1)

            x = self.Vt[i](x)
            
            Sig = Sig.unsqueeze(0)  # 1,259
            sig_repeat = Sig.expand(x.shape[0], -1)  # 30000,259
            x = x * sig_repeat
            
            x = self.U[i](x)

            if i < self.num_layers - 1:
                x = self.relu(x)
                if self.dropout is not None and i in self.dropout:
                    x = F.dropout(x, p=self.dropout_prob, training=self.training)

        x = self.th(x)
        return x
