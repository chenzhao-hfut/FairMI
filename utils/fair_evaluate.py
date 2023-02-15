# -*- coding: utf-8 -*-
"""
@author: LMC_ZC

"""

import pdb
import math
import torch
import numpy as np
import multiprocessing as mp
from .metric import *


def ranking_evaluate(user_emb, item_emb, n_users, n_items, train_u2i, test_u2i, sens=None, indicators='[\'ndcg\', '
                                                                                                      '\'recall\']',
                     topks='[10, 20, 30]', num_workers=4):
    indicators = eval(indicators)
    topks = eval(topks)
    scores = np.matmul(user_emb, item_emb.T)
    perf_info, topk_items = eval_accelerate(scores, n_users, train_u2i, test_u2i, indicators, topks, num_workers)
    perf_info = np.mean(perf_info, axis=0)

    res = {}
    k = 0
    for ind in indicators:
        for topk in topks:
            res[ind + '@' + str(topk)] = perf_info[k]
            k = k + 1

    if sens is not None:
        for topk in topks:
            res['js_dp@' + str(topk)], res['js_eo@' + str(topk)] = js_topk(topk_items, sens, test_u2i, n_users, n_items,
                                                                           topk)

    return res


def eval_accelerate(scores, n_users, train_u2i, test_u2i, indicators, topks, num_workers):
    test_user_set = list(test_u2i.keys())
    perf_info = np.zeros(shape=(len(test_user_set), len(topks) * len(indicators)), dtype=np.float32)
    topk_items = np.zeros(shape=(n_users, max(topks)), dtype=np.int32)

    test_parameters = zip(test_user_set, )

    with mp.Pool(processes=num_workers, initializer=_init_global,
                 initargs=(scores, train_u2i, test_u2i, indicators, topks,)) as pool:
        res = pool.map(test_one_perf, test_parameters)

    for i, one in enumerate(res):
        perf_info[i] = one[0]
        topk_items[one[1][0]] = one[1][1:]

    return perf_info, topk_items


def _init_global(_scores, _train_u2i, _test_u2i, _indicators, _topks):
    global scores, train_u2i, test_u2i, indicators, topks

    scores = _scores
    train_u2i = _train_u2i
    test_u2i = _test_u2i
    indicators = _indicators
    topks = _topks


def test_one_perf(x):
    u_id = x[0]
    score = np.copy(scores[u_id])
    uid_train_pos_items = list(train_u2i[u_id])
    uid_test_pos_items = list(test_u2i[u_id])
    score[uid_train_pos_items] = -np.inf
    score_indices = largest_indices(score, topks)
    res1 = get_perf(score_indices, uid_test_pos_items, topks)
    res2 = get_topks_items(u_id, score_indices)
    return (res1, res2)


def largest_indices(score, topks):
    max_topk = max(topks)
    indices = np.argpartition(score, -max_topk)[-max_topk:]
    indices = indices[np.argsort(-score[indices])]
    return indices


def get_perf(rank, uid_test_pos_items, topks):
    topk_eval = np.zeros(len(indicators) * len(topks), dtype=np.float32)
    k = 0
    for ind in indicators:
        for topk in topks:
            topk_eval[k] = eval(ind)(rank[:topk], uid_test_pos_items)
            k = k + 1
    return topk_eval


def get_topks_items(uid, rank):
    max_topk = max(topks)
    topk_items = rank[:max_topk]
    return np.hstack([np.array(uid, dtype=np.int32), topk_items])
