# -*- coding: utf-8 -*-
"""
@author: LMC_ZC

"""
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

import numpy as np
import pandas as pd
import pickle
from collections import defaultdict
from sklearn.metrics import roc_auc_score

from utils import *
from models import *
from tqdm import tqdm

import scipy.stats as stats

import pdb
import sys

# sensitive attribute encoder
def train_semigcn(gcn, sens, n_users, lr=0.001, num_epochs=1000, device='cpu'):
    sens = torch.tensor(sens).to(torch.long).to(device)
    optimizer = optim.Adam(gcn.parameters(), lr=lr)

    final_loss = 0.0
    for _ in tqdm(range(num_epochs)):
        _, _, su, _ = gcn()
        shuffle_idx = torch.randperm(n_users)
        classify_loss = F.cross_entropy(su[shuffle_idx].squeeze(), sens[shuffle_idx].squeeze())
        optimizer.zero_grad()
        classify_loss.backward()
        optimizer.step()
        final_loss = classify_loss.item()

    print('epoch: %d, classify_loss: %.6f' % (num_epochs, final_loss))


def train_unify_mi(sens_enc, inter_enc, club, dataset, u_sens,
                   n_users, n_items, train_u2i, test_u2i, args):
    optimizer_G = optim.Adam(inter_enc.parameters(), lr=args.lr)
    optimizer_D = optim.Adam(club.parameters(), lr=args.lr)

    train_loader = DataLoader(dataset, shuffle=True, batch_size=args.batch_size, num_workers=args.num_workers)

    e_su, e_si, _, _ = sens_enc.forward()
    e_su = e_su.detach().to(args.device)
    e_si = e_si.detach().to(args.device)
    p_su = conditional_samples(e_su.detach().cpu().numpy())
    p_si = conditional_samples(e_si.detach().cpu().numpy())
    p_su = torch.tensor(p_su).to(args.device)
    p_si = torch.tensor(p_si).to(args.device)

    ex_enc = torch.load(args.pretrain_path)
    
    # please note that, in the phase of training interest encoder, we freeze the parameters of 
    e_xu, e_xi = ex_enc.forward()
    e_xu = e_xu.detach().to(args.device)
    e_xi = e_xi.detach().to(args.device)

    best_perf = 0.0
    for epoch in range(args.num_epochs):
        train_res = {
            'bpr': 0.0,
            'emb': 0.0,
            'lb': 0.0,
            'ub': 0.0,
            'mi': 0.0,
        }

        for uij in train_loader:
            u = uij[0].type(torch.long).to(args.device)
            i = uij[1].type(torch.long).to(args.device)
            j = uij[2].type(torch.long).to(args.device)
            
            # main task
            main_user_emb, main_item_emb = inter_enc.forward()
            bpr_loss, emb_loss = calc_bpr_loss(main_user_emb, main_item_emb, u, i, j)
            emb_loss = emb_loss * args.l2_reg
            
            # auxiliary task
            e_zu, e_zi = inter_enc.forward()
            lb1 = condition_info_nce_for_embeddings(e_xu[torch.unique(u)], e_zu[torch.unique(u)],
                                                   e_su[torch.unique(u)], p_su[torch.unique(u)])
            lb2 = condition_info_nce_for_embeddings(e_xi[torch.unique(i)], e_zi[torch.unique(i)],
                                                   e_si[torch.unique(i)], p_si[torch.unique(i)])
            lb = args.lreg * (lb1 + lb2)
            # our further research found that imposing upper bound constraints on
            # the user-side only gives more stable and better results, so codes has been updated here.
            up = args.ureg * club.forward(e_zu[torch.unique(u)], e_su[torch.unique(u)])
            loss = bpr_loss + emb_loss + lb + up
           
            optimizer_G.zero_grad()
            loss.backward()
            optimizer_G.step()

            train_res['bpr'] += bpr_loss.item()
            train_res['emb'] += emb_loss.item()
            train_res['lb'] += lb.item()
            train_res['ub'] += up.item()

        train_res['bpr'] = train_res['bpr'] / len(train_loader)
        train_res['emb'] = train_res['emb'] / len(train_loader)
        train_res['lb'] = train_res['lb'] / len(train_loader)
        train_res['ub'] = train_res['ub'] / len(train_loader)

        # training for contrastive log-ratio upper bound
        e_zu, e_zi = inter_enc.forward()
        x_samples = e_zu.detach()
        y_samples = e_su.detach()
        for _ in range(args.train_step):
            random_idx = torch.randperm(x_samples.shape[0])
            mi_loss = club.learning_loss(x_samples[random_idx], y_samples[random_idx])
            optimizer_D.zero_grad()
            mi_loss.backward()
            optimizer_D.step()
            train_res['mi'] += mi_loss.item()
        train_res['mi'] = train_res['mi'] / args.train_step

        training_logs = 'epoch: %d, ' % epoch
        for name, value in train_res.items():
            training_logs += name + ':' + '%.6f' % value + ' '
        print(training_logs)

        with torch.no_grad():
            t_user_emb, t_item_emb = inter_enc.forward()
            test_res = ranking_evaluate(
                user_emb=t_user_emb.detach().cpu().numpy(),
                item_emb=t_item_emb.detach().cpu().numpy(),
                n_users=n_users,
                n_items=n_items,
                train_u2i=train_u2i,
                test_u2i=test_u2i,
                sens=u_sens,
                num_workers=args.num_workers)

            p_eval = ''
            for keys, values in test_res.items():
                p_eval += keys + ':' + '[%.6f]' % values + ' '
            print(p_eval)
            
            if best_perf < test_res['ndcg@10']:
                best_perf = test_res['ndcg@10']
                torch.save(model, args.param_path)
                print('save successful')


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='ml_bpr_fairmi',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--bakcbone', type=str, default='bpr')
    parser.add_argument('--dataset', type=str, default='./data/ml-1m/process/process.pkl')
    parser.add_argument('--emb_size', type=int, default=64)
    parser.add_argument('--hidden_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--l2_reg', type=float, default=0.001)
    parser.add_argument('--n_layers', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=2048)
    parser.add_argument('--num_workers', type=int, default=6)
    parser.add_argument('--log_path', type=str, default='logs/bpr_fairmi.txt')
    parser.add_argument('--param_path', type=str, default='param/bpr_fairmi.pth')
    parser.add_argument('--pretrain_path', type=str, default='param/bpr_base.pth')
    parser.add_argument('--lreg', type=float, default=0.1)
    parser.add_argument('--ureg', type=float, default=0.1)
    parser.add_argument('--train_step', type=int, default=50)
    parser.add_argument('--num_epochs', type=int, default=500)
    parser.add_argument('--device', type=str, default='cuda:0')

    args = parser.parse_args()

    sys.stdout = Logger(args.log_path)
    print(args)

    with open(args.dataset, 'rb') as f:
        train_u2i = pickle.load(f)
        train_i2u = pickle.load(f)
        test_u2i = pickle.load(f)
        test_i2u = pickle.load(f)
        train_set = pickle.load(f)
        test_set = pickle.load(f)
        user_side_features = pickle.load(f)
        n_users, n_items = pickle.load(f)

    bprmf = BPRMF(n_users, n_items, args.emb_size, device=args.device)
    u_sens = user_side_features['gender'].astype(np.int32)
    dataset = BPRTrainLoader(train_set, train_u2i, n_items)

    graph = Graph(n_users, n_items, train_u2i)
    norm_adj = graph.generate_ori_norm_adj()

    sens_enc = SemiGCN(n_users, n_items, norm_adj,
                       args.emb_size, args.n_layers, args.device,
                       nb_classes=np.unique(u_sens).shape[0])

    # for interest encoder, we use bprmf as backbone.
    inter_enc = BPRMF(n_users, n_items, args.emb_size, args.device)
    club = CLUBSample(args.emb_size, args.emb_size, args.hidden_size, args.device)
    train_semigcn(sens_enc, u_sens, n_users, device=args.device)
    train_unify_mi(sens_enc, inter_enc, club, dataset, u_sens, n_users, n_items, train_u2i, test_u2i, args)
    sys.stdout = None
