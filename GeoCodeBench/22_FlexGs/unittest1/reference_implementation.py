"""
Reference Implementation for Gumbel_Network
This serves as the ground truth for testing LLM-generated implementations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Gumbel_Network(nn.Module):
    def __init__(self):
        super(Gumbel_Network, self).__init__()
        self.W = 32
        self.pos_emd = nn.Sequential(nn.Linear(3, self.W), nn.ReLU(),nn.Linear(self.W,self.W),nn.ReLU())
        self.rotation_emd = nn.Sequential(nn.Linear(4, self.W), nn.ReLU(),nn.Linear(self.W,self.W),nn.ReLU())
        self.scale_emd = nn.Sequential(nn.Linear(3, self.W), nn.ReLU(),nn.Linear(self.W,self.W),nn.ReLU())
        
        self.time_emd = nn.Sequential(nn.Linear(1, self.W), nn.ReLU(),nn.Linear(self.W,self.W),nn.ReLU())
        
        self.soft_net = nn.Linear(int(self.W * 4), 2)
        self.hard_net = F.gumbel_softmax

    def forward(self, pos, rotation, scale, time, cur_tau=1.0):
        pos_emd = self.pos_emd(pos)
        rotation_emd = self.rotation_emd(rotation)
        scale_emd = self.scale_emd(scale)
        time_emd = self.time_emd(time)
        
        gumbel_input = torch.cat((pos_emd, rotation_emd, scale_emd, time_emd), dim=1)
        soft_output = self.soft_net(gumbel_input)
        hard_output = self.hard_net(soft_output, hard=True, tau=cur_tau)
        soft1 = self.hard_net(soft_output, hard=False, tau=cur_tau)
        
        return soft_output[:, 1], hard_output[:, 1], soft1[:, 1]

