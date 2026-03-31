"""
Reference Implementation for Channel_CTX_fea.forward()
This serves as the ground truth for testing LLM-generated implementations.
"""

import torch
import torch.nn as nn


class Channel_CTX_fea(nn.Module):
    def __init__(self):
        super().__init__()
        self.mean_d0 = nn.Parameter(torch.zeros(size=[1, 64]))
        self.scale_d0 = nn.Parameter(torch.zeros(size=[1, 64]))
        self.prob_d0 = nn.Parameter(torch.zeros(size=[1, 64]))
        self.MLP_d0 = nn.Sequential(
            nn.Linear(64, 64*3),
            nn.LeakyReLU(inplace=True),
            nn.Linear(64*3, 64*3),
            nn.LeakyReLU(inplace=True),
            nn.Linear(64*3, 64*3),
        )
        self.MLP_d1 = nn.Sequential(
            nn.Linear(64*2, 64*3),
            nn.LeakyReLU(inplace=True),
            nn.Linear(64*3, 64*3),
            nn.LeakyReLU(inplace=True),
            nn.Linear(64*3, 64*3),
        )
        self.MLP_d2 = nn.Sequential(
            nn.Linear(64*3, 64*3),
            nn.LeakyReLU(inplace=True),
            nn.Linear(64*3, 64*3),
            nn.LeakyReLU(inplace=True),
            nn.Linear(64*3, 64*3),
        )

    def forward(self, fea_q, to_dec=-1):
        # fea_q: [N, 256]
        NN = fea_q.shape[0]
        d0, d1, d2, d3 = torch.split(fea_q, split_size_or_sections=[64, 64, 64, 64], dim=-1)
        # mean_d0, scale_d0, prob_d0 = torch.zeros_like(d0), torch.zeros_like(d0), torch.zeros_like(d0)  # [N, 64] * 3
        mean_d0, scale_d0, prob_d0 = self.mean_d0.repeat(NN, 1), self.scale_d0.repeat(NN, 1), self.prob_d0.repeat(NN, 1)
        mean_d1, scale_d1, prob_d1 = torch.chunk(self.MLP_d0(d0), chunks=3, dim=-1)  # [N, 64*3] -> [N, 64] * 3
        mean_d2, scale_d2, prob_d2 = torch.chunk(self.MLP_d1(torch.cat([d0, d1], dim=-1)), chunks=3, dim=-1)  # [N, 64*3] -> [N, 64] * 3
        mean_d3, scale_d3, prob_d3 = torch.chunk(self.MLP_d2(torch.cat([d0, d1, d2], dim=-1)), chunks=3, dim=-1)  # [N, 64*3] -> [N, 64] * 3
        mean = torch.cat([mean_d0, mean_d1, mean_d2, mean_d3], dim=-1)
        scale = torch.cat([scale_d0, scale_d1, scale_d2, scale_d3], dim=-1)
        prob = torch.cat([prob_d0, prob_d1, prob_d2, prob_d3], dim=-1)
        if to_dec == 0:
            return mean_d0, scale_d0, prob_d0
        if to_dec == 1:
            return mean_d1, scale_d1, prob_d1
        if to_dec == 2:
            return mean_d2, scale_d2, prob_d2
        if to_dec == 3:
            return mean_d3, scale_d3, prob_d3
        return mean, scale, prob  # [N, 64*4], [N, 64*4], [N, 64*4]



