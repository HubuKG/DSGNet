#!/usr/bin/python3
import torch
import torch.nn as nn
import dgl
import dgl.function as fn
import utils
from utils import get_param
from decoder import ConvE
import torch.nn.functional as F
import numpy as np


class DSGNet(nn.Module):
    def __init__(self, h_dim):
        super().__init__()
        # get params
        self.cfg = utils.get_global_config()
        self.dataset = self.cfg.dataset
        self.device = self.cfg.device
        self.n_ent = utils.DATASET_STATISTICS[self.dataset]['n_ent']
        self.n_rel = utils.DATASET_STATISTICS[self.dataset]['n_rel']
        self.ent_emb = get_param(self.n_ent, h_dim)
        self.kg_n_layer = self.cfg.kg_layer
        self.comp_layers = nn.ModuleList([CompLayer(h_dim) for _ in range(self.kg_n_layer)])

        # relation embedding for aggregation
        self.rel_embs = nn.ParameterList([get_param(self.n_rel * 2, h_dim) for _ in range(self.kg_n_layer)])

        # relation embedding for prediction
        if self.cfg.pred_rel_w:
            self.rel_w = get_param(h_dim * self.kg_n_layer, h_dim)
        else:
            self.pred_rel_emb = get_param(self.n_rel * 2, h_dim)

        self.predictor = ConvE(h_dim, out_channels=self.cfg.out_channel, ker_sz=self.cfg.ker_sz)
        self.bce = nn.BCELoss()
        self.ent_drop = nn.Dropout(self.cfg.ent_drop)
        self.rel_drop = nn.Dropout(self.cfg.rel_drop)
        self.act = nn.Tanh()
        self.L = nn.Linear(h_dim, h_dim)
        self.S = nn.Linear(h_dim, h_dim)
        self.mea_func = Measure_F(h_dim, h_dim, [200] * 2, [200] * 2).to(self.device)
        # self.mlp1 = MLP1(h_dim, [200], h_dim)
        # self.mlp2 = MLP2(h_dim, [200], h_dim)

        self.relu = nn.ReLU()

    def forward(self, h_id, r_id, kg):
        """
        matching computation between query (h, r) and answer t.
        :param h_id: head entity id, (bs, )
        :param r_id: relation id, (bs, )
        :param kg: aggregation graph
        :return: matching score, (bs, n_ent)
        """
        # aggregate embedding
        ent_emb, rel_emb, corr = self.aggragate_emb(kg)

        head = ent_emb[h_id]
        rel = rel_emb[r_id]
        # (bs, n_ent)
        score = self.predictor(head, rel, ent_emb)

        return score, corr

    def loss(self, score, label, corr):
        # (bs, n_ent)
        loss = self.bce(score, label) + 0.01 * corr

        return loss

    def aggragate_emb(self, kg):
        """
        aggregate embedding.
        :param kg:
        :return:
        """
        ent_emb = self.ent_emb
        common = self.S(ent_emb)
        private = self.L(ent_emb)

        corr = 0

        rel_emb_list = []

        for comp_layer, rel_emb in zip(self.comp_layers, self.rel_embs):
            ent_emb, rel_emb = self.ent_drop(ent_emb), self.rel_drop(rel_emb)

            comp_ent_emb1 = comp_layer(kg, common, rel_emb)
            comp_ent_emb2 = comp_layer(kg, private, rel_emb)
            ent_emb = ent_emb + comp_ent_emb1 + comp_ent_emb2
            rel_emb_list.append(rel_emb)
            phi_c, phi_p = self.mea_func(comp_ent_emb1, comp_ent_emb2)
            corr = compute_corr(phi_c, phi_p)

        if self.cfg.pred_rel_w:
            pred_rel_emb = torch.cat(rel_emb_list, dim=1)
            pred_rel_emb = pred_rel_emb.mm(self.rel_w)
        else:
            pred_rel_emb = self.pred_rel_emb

        return ent_emb, pred_rel_emb, corr


class CompLayer(nn.Module):
    def __init__(self, h_dim):
        super().__init__()
        self.cfg = utils.get_global_config()
        self.device = self.cfg.device
        dataset = self.cfg.dataset
        self.n_ent = utils.DATASET_STATISTICS[dataset]['n_ent']
        self.n_rel = utils.DATASET_STATISTICS[dataset]['n_rel']
        self.comp_op = self.cfg.comp_op

        assert self.comp_op in ['add', 'mul']
        self.h_dim = h_dim
        self.neigh_w = get_param(h_dim, h_dim)
        # self.attention_w = utils.get_param(h_dim, 1)
        self.act = nn.Tanh()
        if self.cfg.bn:
            self.bn = torch.nn.BatchNorm1d(h_dim)
            # self.bn0 = torch.nn.BatchNorm1d(1)
        else:
            self.bn = None

    def forward(self, kg, ent_emb, rel_emb):
        assert kg.number_of_nodes() == ent_emb.shape[0]
        assert rel_emb.shape[0] == 2 * self.n_rel

        with kg.local_scope():
            kg.ndata['emb'] = ent_emb
            rel_id = kg.edata['rel_id']
            kg.edata['emb'] = rel_emb[rel_id]

            # neighbor entity and relation composition
            if self.cfg.comp_op == 'add':
                kg.apply_edges(fn.u_add_e('emb', 'emb', 'comp_emb'))
            elif self.cfg.comp_op == 'mul':
                kg.apply_edges(fn.u_mul_e('emb', 'emb', 'comp_emb'))
            else:
                raise NotImplementedError

            # attention
            kg.apply_edges(fn.e_dot_v('comp_emb', 'emb', 'norm'))  # (n_edge, 1)
            kg.edata['norm'] = dgl.ops.edge_softmax(kg, kg.edata['norm'])

            kg.edata['weight'] = kg.edata['norm']
            kg = kg.to('cpu')
            sample_kg = dgl.sampling.select_topk(kg, 15, 'weight', edge_dir='in')
            sample_kg = sample_kg.to(self.device)

            # agg
            sample_kg.edata['comp_emb'] = sample_kg.edata['comp_emb'] * sample_kg.edata['norm']
            sample_kg.update_all(fn.copy_e('comp_emb', 'm'), fn.sum('m', 'neigh'))
            neigh_ent_emb = sample_kg.ndata['neigh']

            neigh_ent_emb = neigh_ent_emb.mm(self.neigh_w)

            if callable(self.bn):
                neigh_ent_emb = self.bn(neigh_ent_emb)

            neigh_ent_emb = self.act(neigh_ent_emb)

        return neigh_ent_emb


def compute_corr(x1, x2):
    # Subtract the mean
    x1_mean = torch.mean(x1, 0, True)
    x1 = x1 - x1_mean
    x2_mean = torch.mean(x2, 0, True)
    x2 = x2 - x2_mean

    # Compute the cross correlation
    sigma1 = torch.sqrt(torch.mean(x1.pow(2)))
    sigma2 = torch.sqrt(torch.mean(x2.pow(2)))
    corr = torch.abs(torch.mean(x1 * x2)) / (sigma1 * sigma2)

    return corr


class MLP(nn.Module):
    def __init__(self, input_d, structure, output_d, dropprob=0.0):
        super(MLP, self).__init__()
        self.net = nn.ModuleList()
        self.dropout = torch.nn.Dropout(dropprob)
        struc = [input_d] + structure + [output_d]

        for i in range(len(struc) - 1):
            self.net.append(nn.Linear(struc[i], struc[i + 1]))

    def forward(self, x):
        for i in range(len(self.net) - 1):
            x = F.relu(self.net[i](x))
            x = self.dropout(x)

        # For the last layer
        y = self.net[-1](x)

        return y


# class MLP1(nn.Module):
#     def __init__(self, input_d, structure, output_d, dropprob=0.0):
#         super(MLP1, self).__init__()
#         self.net = nn.ModuleList()
#         self.dropout = torch.nn.Dropout(dropprob)
#         struc = [input_d] + structure + [output_d]
#
#         for i in range(len(struc) - 1):
#             self.net.append(nn.Linear(struc[i], struc[i + 1]))
#
#     def forward(self, x):
#         for i in range(len(self.net) - 1):
#             x = F.relu(self.net[i](x))
#             x = self.dropout(x)
#
#         # For the last layer
#         y = self.net[-1](x)
#
#         return y
#
#
# class MLP2(nn.Module):
#     def __init__(self, input_d, structure, output_d, dropprob=0.0):
#         super(MLP2, self).__init__()
#         self.net = nn.ModuleList()
#         self.dropout = torch.nn.Dropout(dropprob)
#         struc = [input_d] + structure + [output_d]
#
#         for i in range(len(struc) - 1):
#             self.net.append(nn.Linear(struc[i], struc[i + 1]))
#
#     def forward(self, x):
#         for i in range(len(self.net) - 1):
#             x = F.relu(self.net[i](x))
#             x = self.dropout(x)
#
#         # For the last layer
#         y = self.net[-1](x)
#
#         return y


# measurable functions \phi and \psi
class Measure_F(nn.Module):
    def __init__(self, view1_dim, view2_dim, phi_size, psi_size, latent_dim=1):
        super(Measure_F, self).__init__()
        self.phi = MLP(view1_dim, phi_size, latent_dim)
        self.psi = MLP(view2_dim, psi_size, latent_dim)

    def forward(self, x1, x2):
        y1 = self.phi(x1)
        y2 = self.psi(x2)
        return y1, y2





