# -*- coding: utf-8 -*-
"""
@author: LMC_ZC

"""


import torch
import torch.nn as nn


class BPRMF(nn.Module):
    def __init__(self, n_users, n_items, emb_size, device):
        super(BPRMF, self).__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.emb_size = emb_size
        self.device = device

        self._init_embeddings()

    def _init_embeddings(self, ):
        self.embeddings = nn.ModuleDict()
        self.embeddings['user_embeddings'] = nn.Embedding(self.n_users, self.emb_size).to(self.device)
        self.embeddings['item_embeddings'] = nn.Embedding(self.n_items, self.emb_size).to(self.device)
        nn.init.xavier_uniform_(self.embeddings['user_embeddings'].weight)
        nn.init.xavier_uniform_(self.embeddings['item_embeddings'].weight)

    def forward(self, ):
        return self.embeddings['user_embeddings'].weight, self.embeddings['item_embeddings'].weight
