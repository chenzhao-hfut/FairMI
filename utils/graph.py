# -*- coding: utf-8 -*-
"""

@author: LMC_ZC
"""


import torch


class Graph(object):
    def __init__(self, n_users, n_items, train_u2i):
        self.n_users = n_users
        self.n_items = n_items
        self.train_u2i = train_u2i

    def to_node(self, train_u2i):
        node1, node2 = [], []

        for i, j in train_u2i.items():
            node1.extend([i] * len(j))
            node2.extend(j)

        node1 = torch.tensor(node1, dtype=torch.long)
        node2 = torch.tensor(node2, dtype=torch.long)
        return node1, node2

    def to_edge(self, train_u, train_i):
        row = torch.cat([train_u, train_i + self.n_users])
        col = torch.cat([train_i + self.n_users, train_u])
        edge_index = torch.stack([row, col]).to(torch.long)
        edge_weight = torch.ones_like(row).to(torch.float32)
        return edge_index, edge_weight

    def generate_ori_norm_adj(self):
        user_nodes, item_nodes = self.to_node(self.train_u2i)
        edge_index, edge_weight = self.to_edge(user_nodes, item_nodes)
        norm_adj = self.generate(edge_index, edge_weight)
        return norm_adj

    def generate(self, edge_index, edge_weight):
        edge_index, edge_weight = self.add_self_loop(edge_index, edge_weight)
        edge_index, edge_weight = self.norm(edge_index, edge_weight)
        return self.mat(edge_index, edge_weight)

    def add_self_loop(self, edge_index, edge_weight):
        """ add self-loop """

        loop_index = torch.arange(0, self.num_nodes, dtype=torch.long)
        loop_index = loop_index.unsqueeze(0).repeat(2, 1)
        loop_weight = torch.ones(self.num_nodes, dtype=torch.float32)
        edge_index = torch.cat([edge_index, loop_index], dim=-1)
        edge_weight = torch.cat([edge_weight, loop_weight], dim=-1)

        return edge_index, edge_weight

    def norm(self, edge_index, edge_weight):
        """ D^{-1/2} * A * D^{-1/2}  """
        row, col = edge_index[0], edge_index[1]
        deg = torch.zeros(self.num_nodes, dtype=torch.float32)
        deg = deg.scatter_add(0, col, edge_weight)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
        edge_weight = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

        return edge_index, edge_weight

    @property
    def num_nodes(self):
        return self.n_users + self.n_items

    def mat(self, edge_index, edge_weight):
        return torch.sparse_coo_tensor(edge_index, edge_weight, torch.Size([self.num_nodes, self.num_nodes]))