#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: Haojun Chen
@Contact: chjchjchjchjchj2001@gmail.com
"""

import os
import sys
import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.neighbors import NearestNeighbors
import ipdb

class BaseGCN(nn.Module):
    def __init__(self, N=1024):
        super(BaseGCN, self).__init__()
        self.N = N

    def forward(self, x):
        raise NotImplementedError

    def get_adj(self, x):
        """
        Calculates adjacency matrices based on pairwise distances between points.

        Args:
            x: Input tensor of shape (B, N, C), where B is the batch size, N is the number of points, and C is the number of features.

        Returns:
            Adjacency matrices of shape (B, N, N).
        """
        B, N, _ = x.size()
        distances = torch.cdist(x, x, p=2) # (B, N, N) Pairwise distances between points using Euclidean distance metric
        _, indices = torch.topk(distances, k=self.k, dim=-1, largest=False) # _: (B, N, N), sorted distances ; indices: (B, N, k), index of the knn for each point
        adjacency_matrices = torch.zeros(B, N, N, device=x.device)
        adjacency_matrices.scatter_(2, indices, 1) # modifies the adjacency matrices by setting the indices of the k nearest neighbors to 1
        return adjacency_matrices

    def process_graph(self, adjs):
        """
        Processes the adjacency matrices to obtain normalized adjacency matrices.

        Args:
            adjs: Input adjacency matrices of shape (B, N, N).

        Returns:
            According of L=I_N-D^{-\frac{1}{2}}AD^{-\frac{1}{2}} in https://arxiv.org/pdf/1609.02907.pdf
            Normalized adjacency matrices of shape (B, N, N).
        """
        B, N, _ = adjs.size()
        degrees = torch.sum(adjs, dim=2) # (B, N)
        D = torch.diag_embed(degrees) # (B, N, N)
        D_sqrt_inv = torch.sqrt(torch.reciprocal(D)) # (B, N, N) D^{-\frac{1}{2}}
        D_sqrt_inv[torch.isinf(D_sqrt_inv)] = 0.0 # Replaces any infinite values in D_sqrt_inv with 0.0

        normalized_adj_matrices = torch.eye(N, device=D.device).unsqueeze(0).repeat(B, 1, 1) - torch.bmm(torch.bmm(D_sqrt_inv, adjs), D_sqrt_inv.transpose(1, 2)) # I_N-D^{-\frac{1}{2}}AD^{-\frac{1}{2}}

        return normalized_adj_matrices

    def address_overfitting_graph(self, adjs):
        """
        Addresses overfitting in adjacency matrices by adding self-loops and performing normalization.

        Args:
            adjs: Input adjacency matrices of shape (B, N, N).

        Returns:
            According to \tilde{D} = \tilde{D}^{-\frac12}\tilde{A}\tilde{D}^{-\frac12} in https://arxiv.org/pdf/1609.02907.pdf
            New adjacency matrices with self-loops and normalization, of shape (B, N, N).
        """
        B, N, _ = adjs.size()
        indentity = torch.eye(N, device=adjs.device).unsqueeze(0).repeat(B, 1, 1)
        adjs_hat = adjs + indentity
        D_hat = torch.diag_embed(torch.sum(adjs_hat, dim=2))
        D_hat_sqrt_inv = torch.sqrt(torch.reciprocal(D_hat))
        D_hat_sqrt_inv[torch.isinf(D_hat_sqrt_inv)] = 0.0
        new_matrix = torch.bmm(torch.bmm(D_hat_sqrt_inv, adjs_hat), D_hat_sqrt_inv)

        return new_matrix

class BatchNormWrapper(nn.Module):
    def __init__(self, features):
        super(BatchNormWrapper, self).__init__()
        self.batchnorm = nn.BatchNorm1d(features)

    def forward(self, data):
        batch_size, num_points, _ = data.size()
        reshaped_data = data.view(-1, data.size(2))
        normalized_data = self.batchnorm(reshaped_data)
        normalized_data = normalized_data.view(batch_size, num_points, -1)
        return normalized_data

class GCNResidualBlock(nn.Module):
    def __init__(self, args, in_features, hidden_dim, out_dim):
        super(GCNResidualBlock, self).__init__()
        self.gcn1 = GCNLayer(args, in_features, hidden_dim)
        self.gcn2 = GCNLayer(args, hidden_dim, out_dim)
        self.bn = nn.BatchNorm1d(out_dim)
        self.dropout = args.dropout

    def forward(self, x, adj):
        """
        Args:
            x: (B, N, C) - Batch of input features for each node
            adj: (B, N, N) - Batch of adjacency matrices
        Returns:
            (B, N, out_dim)
        """
        h = self.gcn1(x, adj) # (B, N, hidden_dim)
        h = F.relu(h)
        h = self.gcn2(h, adj) # (B, N, out_dim)
        B, N, C = h.size()
        h = h.view(-1, C)
        h = self.bn(h) # (B*N, out_dim)
        h = h.view(B, N, C) # (B, N, out_dim)
        return F.relu(h + x)

class GCNResNet(BaseGCN):
    def __init__(self, args, N=1024):
        super(GCNResNet, self).__init__()
        in_features, hidden_features, out_features, num_blocks = args.res_in, args.res_hid, args.res_out, args.res_num_blocks
        self.initial_gcn = GCNLayer(args, in_features, hidden_features)
        self.bn = nn.BatchNorm1d(hidden_features)
        if args.use_polynomial:
            self.res_blocks = nn.ModuleList([
                GCNPolynomialBlock(args, hidden_features, hidden_features, hidden_features)
                for _ in range(num_blocks)
            ])
        else:
            self.res_blocks = nn.ModuleList([
                GCNResidualBlock(args, hidden_features, hidden_features, hidden_features)
                for _ in range(num_blocks)
            ])
        self.k = args.k
        self.address_overfitting = args.address_overfitting
        self.final_gcn = GCNLayer(args, hidden_features, out_features)

        self.linear1 = nn.Linear(in_features, hidden_features)
        self.linear2 = nn.Linear(hidden_features, 128)
        self.bn2 = BatchNormWrapper(features=128)

        self.linear3 = nn.Linear(128, 512)
        self.bn3 = BatchNormWrapper(features=512)

        self.linear4 = nn.Linear(512, 1024)
        self.bn4 = BatchNormWrapper(features=1024)

        self.linear5 = nn.Linear(1024, 1024)
        self.bn5 = nn.BatchNorm1d(1024)

        self.linear6 = nn.Linear(1024, 512)
        self.bn6 = nn.BatchNorm1d(512)

        self.linear7 = nn.Linear(512, out_features)
        self.LinearLayers1 = nn.Sequential(self.linear2, self.bn2, nn.ReLU(), self.linear3, self.bn3, nn.ReLU(), self.linear4, self.bn4, nn.ReLU())
        self.pool1 = nn.AdaptiveAvgPool1d(1)
        self.LinearLayers2 = nn.Sequential(self.linear5, self.bn5, nn.ReLU(), self.linear6, self.bn6, nn.ReLU(), self.linear7)

        
    def forward(self, x):
        adjs = self.get_adj(x)

        if self.address_overfitting:
            A = self.address_overfitting_graph(adjs).float()
        else:
            A = self.process_graph(adjs).float()

        h = self.initial_gcn(x, A)
        h = F.relu(h)
        B, N, C = h.size()
        h = h.view(-1, C)
        h = self.bn(h)
        h = h.view(B, N, C)
        
        for block in self.res_blocks:
            h = block(h, A)
        
        h = self.LinearLayers1(h)
        h = self.pool1(h.transpose(1, 2)).squeeze() # (B, N, out_features) -> (B, out_features)
        h = self.LinearLayers2(h)

        # h = self.final_gcn(h, A)
        return h

class GCNPolynomialBlock(nn.Module):
    def __init__(self, args, in_features, hidden_dim, out_dim):
        super(GCNPolynomialBlock, self).__init__()
        self.gcn1 = GCNLayer(args, in_features, hidden_dim)
        self.gcn2 = GCNLayer(args, hidden_dim, out_dim)
        self.bn = nn.BatchNorm1d(out_dim)
        self.dropout = args.dropout

    def forward(self, x, adj):
        x = self.gcn1(x, adj)
        x = F.relu(x)
        x = self.gcn2(x, adj)
        B, N, C = x.size()
        x = x.view(-1, C)
        x = self.bn(x)
        x = x.view(B, N, C)
        return F.relu(x)

class GCNPolynomial(BaseGCN):
    def __init__(self, args, N=1024):
        super(GCNPolynomial, self).__init__()
        N = args.num_points
        in_features, hidden_features, out_features = args.res_in, args.res_hid, args.res_out
        self.times = args.f_times
        self.initial_gcn = GCNLayer(args, args.res_in, args.res_hid)
        # self.mid_gcn = GCNLayer(args, args.res_hid, args.res_hid)
        self.midblock = GCNPolynomialBlock(args, args.res_hid, args.res_hid, args.res_hid)
        self.bn = BatchNormWrapper(features=args.res_hid)
        self.k = args.k
        self.address_overfitting = args.address_overfitting
        self.final_gcn = GCNLayer(args, hidden_features, out_features)

        self.linear1 = nn.Linear(in_features, hidden_features)
        self.linear2 = nn.Linear(hidden_features, 128)
        self.bn2 = BatchNormWrapper(features=128)

        self.linear3 = nn.Linear(128, 512)
        self.bn3 = BatchNormWrapper(features=512)

        self.linear4 = nn.Linear(512, 1024)
        self.bn4 = BatchNormWrapper(features=1024)

        self.linear5 = nn.Linear(1024, 1024)
        self.bn5 = nn.BatchNorm1d(1024)

        self.linear6 = nn.Linear(1024, 512)
        self.bn6 = nn.BatchNorm1d(512)

        self.linear7 = nn.Linear(512, out_features)
        self.LinearLayers1 = nn.Sequential(self.linear2, self.bn2, nn.ReLU(), self.linear3, self.bn3, nn.ReLU(), self.linear4, self.bn4, nn.ReLU())
        self.pool1 = nn.AdaptiveAvgPool1d(1)
        self.LinearLayers2 = nn.Sequential(self.linear5, self.bn5, nn.ReLU(), self.linear6, self.bn6, nn.ReLU(), self.linear7)
    
    def forward(self, x):
        adjs = self.get_adj(x)

        if self.address_overfitting:
            A = self.address_overfitting_graph(adjs).float()
        else:
            A = self.process_graph(adjs).float()
        
        all_features = []
        for i in range(self.times+1):
            if i == 0:
                # x = self.linear1(x)
                # tmp = x.clone()
                # all_features.append(x)
                x = F.relu(self.initial_gcn(x, A))
                x = self.bn(x)
            else:
                tmp = x.clone()
                all_features.append(tmp)
                # x = F.relu(self.mid_gcn(x, A))
                x = self.midblock(x, A)
                x = self.bn(x)
                if i == self.times:
                    all_features.append(x)
        h = sum(all_features)
        h = self.LinearLayers1(h)
        h = self.pool1(h.transpose(1, 2)).squeeze()
        h = self.LinearLayers2(h)

        return h
    
            
    

# HJ
class GCNLayer(nn.Module):
    def __init__(self, args, input_dim, output_dim) -> None:
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.dropout = args.dropout
        # self.dp1 = nn.Dropout(p=args.dropout)

    def forward(self, x, adj):
        output = self.linear(x)
        output = F.relu(torch.bmm(adj, output))
        # output = self.dp1(output)
        return output

class GCN(BaseGCN):
    def __init__(self, args, in_c=3, hid_c=64, out_c=40, N=1024) -> None:
        super(GCN, self).__init__()
        N = args.num_points
        self.linear1 = nn.Linear(in_c, hid_c)
        self.linear2 = nn.Linear(hid_c, 128)
        self.bn2 = BatchNormWrapper(features=128)

        self.linear3 = nn.Linear(128, 512)
        self.bn3 = BatchNormWrapper(features=512)

        self.linear4 = nn.Linear(512, 1024)
        self.bn4 = BatchNormWrapper(features=1024)

        self.linear5 = nn.Linear(1024, 1024)
        self.bn5 = nn.BatchNorm1d(1024)

        self.linear6 = nn.Linear(1024, 512)
        self.bn6 = nn.BatchNorm1d(512)

        self.linear7 = nn.Linear(512, out_c)
        self.gcn_layers = args.gcn_layers
        self.act = nn.ReLU()
        self.dropout = args.dropout
        self.k = args.k
        self.address_overfitting = args.address_overfitting
        self.gc_layers = nn.ModuleList([GCNLayer(args, in_c, hid_c)])
        for _ in range(self.gcn_layers - 1):
            self.gc_layers.append(GCNLayer(args, hid_c, hid_c))
        self.LinearLayers1 = nn.Sequential(self.linear2, self.bn2, nn.ReLU(), self.linear3, self.bn3, nn.ReLU(), self.linear4, self.bn4, nn.ReLU())
        self.pool1 = nn.MaxPool1d(N)
        self.LinearLayers2 = nn.Sequential(self.linear5, self.bn5, nn.ReLU(), self.linear6, self.bn6, nn.ReLU(), self.linear7)
    
    def create_batchnorm(self, features):
        return lambda data: nn.BatchNorm1d(features)(data.view(-1, features)).view(data.size())

    def forward(self, x):
        # x = x.permute(0, 2, 1)
        adjs = self.get_adj(x)
        # ipdb.set_trace() 
        if self.address_overfitting:
            A = self.address_overfitting_graph(adjs).float()
        else:
            A = self.process_graph(adjs).float()

        for layer in self.gc_layers:
            x = layer(x, A)
            x = F.dropout(x, self.dropout, training=self.training)
        # ipdb.set_trace()
        x = self.LinearLayers1(x)
        x = self.pool1(x.transpose(1, 2)).squeeze()
        x = self.LinearLayers2(x)
        # output = F.log_softmax(x, dim=-1)
        return x      


