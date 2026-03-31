
import torch.nn as nn
import torch
import torch.nn.functional as F
import  numpy as np
import time
from utils import embedder
from torch.autograd import grad


class Dict(nn.Module):
    def __init__(
        self,
        bias,
        U,
        Sigma,
        Vt,
        latent_in = [4],
        dropout_prob = 0.0,
        dropout = [0, 1, 2, 3, 4, 5, 6, 7],
        positional_enc = False,
        n_positional_freqs = 8,
    ):
        super(Dict, self).__init__()
        self.U_w = U
        self.U = nn.ModuleList([])
        self.Sigma = nn.ParameterList([])
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

    #     线性层参数初始化
    #     print(len(self.U),len(self.Sigma),len(self.Vt))

        for i in range(len(self.U)):
            self.U[i].weight = nn.Parameter(U[i])
            self.U[i].bias = nn.Parameter(bias[i])
            self.Vt[i].weight = nn.Parameter(Vt[i])

        if positional_enc:
            self.n_positional_freqs = n_positional_freqs
            self.pos_embedder, pos_embedder_out_dim = embedder.get_embedder_nerf(
                self.n_positional_freqs, input_dims=3, i=0
            )

    # def get_diag(self, sigma, width, length):
    #     sigma_m = torch.diag_embed(sigma)
    #     if length < width:
    #         zeros = torch.zeros(length, width - length).cuda()
    #         sigma_m = torch.cat((sigma_m, zeros), axis=1)
    #     elif length > width:
    #         zeros = torch.zeros(length - width, width).cuda()
    #         sigma_m = torch.cat((sigma_m, zeros), axis=0)
    #     return sigma_m

    def forward(self, input):
        # Sigma_m = []
        # for i, sigma in enumerate(self.Sigma):
        #     Sigma_m.append(self.get_diag(sigma, self.U_w[i].shape[1], self.Vt_w[i].shape[0]))
        #     # print(i,sigma_m.shape)

        if hasattr(self, "pos_embedder"):
            xyz = input[:, -3:]
            ****EMPTY****
        else:
            x = input
        # print("self.num_layers", self.num_layers)

        for i, Sig in enumerate(self.Sigma):

            if i in self.latent_in:
                if hasattr(self, "pos_embedder"):
                    x = torch.cat([x, input_embed], 1)
                    # print(x.shape)
                else:
                    x = torch.cat([x, input], 1)

            # print(x)
            # print("!!!",i, x.shape)
            x = self.Vt[i](x)
            
            ****EMPTY****

            # print(i, x.shape, x)
            if i < self.num_layers - 1:
                x = self.relu(x)
                # print("after relu",i,x)
                if self.dropout is not None and i in self.dropout:
                    x = F.dropout(x, p=self.dropout_prob, training=self.training)
                    # print("after dropout : ", i, x.shape, x)
            # print("final output",i, x.shape,x)

        x = self.th(x)
        return x


