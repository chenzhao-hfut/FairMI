# -*- coding: utf-8 -*-
"""
@author: LMC_ZC

"""

import torch
import torch.nn as nn


class LightGCN(nn.Module):

    def __init__(self, n_users, n_items, norm_adj, emb_size, n_layers, device):
        super(LightGCN, self).__init__()

        self.n_users = n_users
        self.n_items = n_items
        self.norm_adj = norm_adj.to(device)
        self.emb_size = emb_size
        self.n_layers = n_layers
        self.device = device
        self._init_embeddings()

    def _init_embeddings(self, ):
        self.embeddings = nn.ModuleDict()
        self.embeddings['user_embeddings'] = nn.Embedding(self.n_users, self.emb_size).to(self.device)
        self.embeddings['item_embeddings'] = nn.Embedding(self.n_items, self.emb_size).to(self.device)
        nn.init.xavier_uniform_(self.embeddings['user_embeddings'].weight)
        nn.init.xavier_uniform_(self.embeddings['item_embeddings'].weight)

    def forward(self, ):
        return self.propagate(
            self.norm_adj,
            self.embeddings['user_embeddings'].weight,
            self.embeddings['item_embeddings'].weight)

    def propagate(self, adj, user_emb, item_emb):
        ego_embeddings = torch.cat([user_emb, item_emb], dim=0)
        all_embeddings = [ego_embeddings]

        for k in range(1, self.n_layers+1):
            if adj.is_sparse is True:
                ego_embeddings = torch.sparse.mm(adj, ego_embeddings)
            else:
                ego_embeddings = torch.mm(adj, ego_embeddings)
            all_embeddings += [ego_embeddings]

        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = torch.mean(all_embeddings, dim=1)
        u_g_embeddings, i_g_embeddings = torch.split(all_embeddings, [self.n_users, self.n_items], dim=0)

        return u_g_embeddings, i_g_embeddings
