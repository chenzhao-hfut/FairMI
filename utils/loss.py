# -*- coding: utf-8 -*-
"""

@author: LMC_ZC
"""

import pdb
import torch
import torch.nn.functional as F


def bpr_loss(user_emb, pos_emb, neg_emb):
    pos_score = torch.sum(user_emb * pos_emb, dim=1)
    neg_score = torch.sum(user_emb * neg_emb, dim=1)
    mf_loss = torch.mean(F.softplus(neg_score - pos_score))
    emb_loss = (1 / 2) * (user_emb.norm(2).pow(2) +
                          pos_emb.norm(2).pow(2) +
                          neg_emb.norm(2).pow(2)) / user_emb.shape[0]
    return mf_loss, emb_loss


def calc_bpr_loss(user_emb, item_emb, u, i, j):
    batch_user_emb = user_emb[u]
    batch_pos_item_emb = item_emb[i]
    batch_neg_item_emb = item_emb[j]

    mf_loss, emb_loss = bpr_loss(batch_user_emb, batch_pos_item_emb, batch_neg_item_emb)
    return mf_loss, emb_loss


def mse_loss(user_emb, pos_emb, r):
    r_hat = (user_emb * pos_emb).sum(dim=-1)
    mf_loss = F.mse_loss(r_hat.squeeze(), r)
    emb_loss = (1 / 2) * (user_emb.norm(2).pow(2) +
                          pos_emb.norm(2).pow(2)) / user_emb.shape[0]
    return mf_loss, emb_loss


def calc_mse_loss(user_emb, item_emb, u, i, r):
    batch_user_emb = user_emb[u]
    batch_pos_item_emb = item_emb[i]
    mf_loss, emb_loss = mse_loss(batch_user_emb, batch_pos_item_emb, r)
    return mf_loss, emb_loss


def condition_info_nce_for_embeddings(x, z, s, p, alpha=0.1, tau=0.5):
    N = x.shape[0]

    x_norm = F.normalize(x)
    y_norm = F.normalize(z + alpha * s)

    pos_score = torch.sum(x_norm * y_norm, dim=1)
    pos_score = torch.exp(pos_score / tau)

    neg_score = torch.zeros(N, dtype=torch.float32).to(x.device)

    for cat in set(p.tolist()):
        x_given = x_norm[p == cat]
        y_given = y_norm[p == cat]

        t = x_given @ y_given.T
        t = torch.sum(torch.exp(t / tau), dim=1)
        neg_score[p == cat] = t

    cl_loss = -torch.log(pos_score / neg_score).mean()
    return cl_loss
