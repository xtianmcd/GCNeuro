from __future__ import division
from __future__ import print_function
import math
import numpy as np
import scipy.sparse as sp
import time
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import chisquare as X2
from utils import *

class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.zeros(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, input, adj):
        h = torch.mm(input, self.W)
        N = h.size()[0]

        a_input = torch.cat([h.repeat(1, N).view(N * N, -1), h.repeat(N, 1)], dim=1).view(N, -1, 2 * self.out_features)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))

        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, h)
        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'

class GCN(nn.Module):
    def __init__(self, nin, nhidd, nout, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nin, nhidd)
        self.gc2 = GraphConvolution(nhidd, nout)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return x #, dim=1)

class GAT(nn.Module):
    def __init__(self, nin, nhidd, nout, dropout, nheads, alpha):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout

        self.attentions = [GraphAttentionLayer(nin, nhidd, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphAttentionLayer(nhidd * nheads, nout, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        return F.log_softmax(x, dim=-1)

class GCNetwork(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, nheads, alpha):
        super(GCNetwork,self).__init__()

        self.subgc = GCN(nfeat,int(nhid**2),int((nhid**2)/2),dropout)
        # self.p_gcn = GCN(int((nhid**2)/2),nhid,nclass,dropout)
        self.gat = GAT(int((nhid**2)/2), nhid, nclass, dropout, nheads, alpha)

    def forward(self, f, adj):
        f1 = torch.tanh(self.subgc(f[0], adj))
        f2 = torch.tanh(self.subgc(f[1], adj))
        f3 = torch.tanh(self.subgc(f[2], adj))
        f4 = torch.tanh(self.subgc(f[3], adj))

        xf = torch.cat([f1.view(f1.size()[0],f1.size()[1],1),
                        f2.view(f1.size()[0],f1.size()[1],1),
                        f3.view(f1.size()[0],f1.size()[1],1),
                        f4.view(f1.size()[0],f1.size()[1],1)],-1)

        mp = nn.MaxPool1d((4))
        xp = mp(xf).view(f1.size())

        # xo = F.log_softmax(self.p_gcn(xp,adj),dim=1)
        xo = self.gat(xp,adj.to_dense())
        o = torch.mean(xo, dim=0)
        return o
