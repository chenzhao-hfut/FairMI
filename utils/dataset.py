# -*- -*- coding: utf-8 -*-
"""
@author: LMC_ZC
"""

import torch
import random


class BPRTrainLoader(torch.utils.data.Dataset):

    def __init__(self, train_set, train_u2i, n_items):
        self.train_set = train_set
        self.train_u2i = train_u2i
        self.all_items = list(range(0, n_items))

    def __getitem__(self, index):
        user = self.train_set['userid'][index]
        pos = self.train_set['itemid'][index]
        neg = random.choice(self.all_items)

        while neg in self.train_u2i[user]:
            neg = random.choice(self.all_items)

        return [user, pos, neg]

    def __len__(self):
        return self.train_set['userid'].shape[0]
