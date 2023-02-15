# -*- coding: utf-8 -*-
"""
@author: LMC_ZC

"""


import torch
import torch.nn as nn
from .lightgcn import LightGCN


class SemiGCN(nn.Module):
    def __init__(self, n_users, n_items, norm_adj, emb_size, n_layers, device, nb_classes):
        super(SemiGCN, self).__init__()
        self.body = LightGCN(n_users, n_items, norm_adj, emb_size, n_layers, device)
        self.fc = nn.Linear(emb_size, nb_classes)
        self.to(device)

    def forward(self, ):
        e_su, e_si = self.body()
        su = self.fc(e_su)
        si = self.fc(e_si)
        return e_su, e_si, su, si

