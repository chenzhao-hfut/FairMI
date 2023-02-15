# -*- coding: utf-8 -*-
"""

@author : LMC_ZC
"""

import math
import scipy
import numpy as np
from collections import defaultdict


def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())


def recall(ranked_list, ground_list):
    hits = 0
    for i in range(len(ranked_list)):
        id = ranked_list[i]
        if id in ground_list:
            hits += 1
    rec = hits / (1.0 * len(ground_list))
    return rec


def ndcg(ranked_list, ground_truth):
    dcg = 0
    idcg = IDCG(len(ground_truth))
    for i in range(len(ranked_list)):
        id = ranked_list[i]
        if id not in ground_truth:
            continue
        rank = i + 1
        dcg += 1 / math.log(rank + 1, 2)
    return dcg / idcg


def IDCG(n):
    idcg = 0
    for i in range(n):
        idcg += 1 / math.log(i + 2, 2)
    return idcg


def js_topk(topk_items, sens, test_u2i, n_users, n_items, topk):
    rank_topk_items = np.zeros((n_users, n_items), dtype=np.int32)
    truth_rank_topk_items = np.zeros((n_users, n_items), dtype=np.int32)
    test_topk_items = topk_items.tolist()
    for uid in range(n_users):
        rank_topk_items[uid][test_topk_items[uid][:topk]] = 1
        truth_rank_topk_items[uid][test_u2i[uid]] = 1

    truth_rank_topk_items = truth_rank_topk_items & rank_topk_items

    index1 = (sens == 1)
    index2 = ~index1

    rank_dis1 = np.sum(rank_topk_items[index1], axis=0)
    rank_dis2 = np.sum(rank_topk_items[index2], axis=0)
    truth_rank_dis1 = np.sum(truth_rank_topk_items[index1], axis=0)
    truth_rank_dis2 = np.sum(truth_rank_topk_items[index2], axis=0)

    rank_js_distance = scipy.spatial.distance.jensenshannon(rank_dis1, rank_dis2)
    truth_rank_js_distance = scipy.spatial.distance.jensenshannon(truth_rank_dis1, truth_rank_dis2)

    return rank_js_distance, truth_rank_js_distance
