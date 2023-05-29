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
def knn(x, k):
    inner = -2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
 
    idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (batch_size, num_points, k)
    return idx


def get_graph_feature(x, k=20, idx=None):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(x, k=k)   # (batch_size, num_points, k)
    device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points

    idx = idx + idx_base

    idx = idx.view(-1)
 
    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()   # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size*num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims) 
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    
    feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 1, 2).contiguous()
  
    return feature


class PointNet(nn.Module):
    def __init__(self, args, output_channels=40):
        super(PointNet, self).__init__()
        self.args = args
        self.conv1 = nn.Conv1d(3, 64, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.conv3 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.conv4 = nn.Conv1d(64, 128, kernel_size=1, bias=False)
        self.conv5 = nn.Conv1d(128, args.emb_dims, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(args.emb_dims)
        self.linear1 = nn.Linear(args.emb_dims, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout()
        self.linear2 = nn.Linear(512, output_channels)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.adaptive_max_pool1d(x, 1).squeeze()
        x = F.relu(self.bn6(self.linear1(x)))
        x = self.dp1(x)
        x = self.linear2(x)
        return x


class DGCNN(nn.Module):
    def __init__(self, args, output_channels=40):
        super(DGCNN, self).__init__()
        self.args = args
        self.k = args.k
        
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm1d(args.emb_dims)

        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64*2, 128, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(128*2, 256, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv1d(512, args.emb_dims, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.linear1 = nn.Linear(args.emb_dims*2, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=args.dropout)
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=args.dropout)
        self.linear3 = nn.Linear(256, output_channels)

    def forward(self, x):
        batch_size = x.size(0)
        x = get_graph_feature(x, k=self.k)
        x = self.conv1(x)
        x1 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x1, k=self.k)
        x = self.conv2(x)
        x2 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x2, k=self.k)
        x = self.conv3(x)
        x3 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x3, k=self.k)
        x = self.conv4(x)
        x4 = x.max(dim=-1, keepdim=False)[0]

        x = torch.cat((x1, x2, x3, x4), dim=1)

        x = self.conv5(x)
        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)
        x = torch.cat((x1, x2), 1)

        x = F.leaky_relu(self.bn6(self.linear1(x)), negative_slope=0.2)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2)
        x = self.dp2(x)
        x = self.linear3(x)
        return x

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
        h = self.gcn1(x, adj)
        h = F.relu(h)
        h = self.gcn2(h, adj)
        B, N, C = h.size()
        h = h.view(-1, C)
        h = self.bn(h)
        h = h.view(B, N, C)
        # h = F.relu(h + x)
        # output = F.dropout(h, self.dropout, training=self.training)
        # return output  # residual
        return F.relu(h + x)

class GCNResNet(nn.Module):
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
        # self.dp = nn.Dropout(p=args.dropout)
        # self.LinearLayers1 = nn.Sequential(self.linear2, self.bn2, nn.ReLU(), self.dp, self.linear3, self.bn3, nn.ReLU(), self.dp, self.linear4, self.bn4, nn.ReLU(), self.dp)
        # self.pool1 = nn.MaxPool1d(N)
        # self.LinearLayers2 = nn.Sequential(self.linear5, self.bn5, nn.ReLU(), self.dp, self.linear6, self.bn6, nn.ReLU(), self.dp, self.linear7)
        # self.dp = nn.Dropout(p=args.dropout)
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
        h = self.pool1(h.transpose(1, 2)).squeeze()
        h = self.LinearLayers2(h)

        # h = self.final_gcn(h, A)
        return h

    def get_adj(self, x):
        B, N, _ = x.size()
        distances = torch.cdist(x, x, p=2) # (B, N, N)
        _, indices = torch.topk(distances, k=self.k, dim=-1, largest=False)
        adjacency_matrices = torch.zeros(B, N, N, device=x.device)
        adjacency_matrices.scatter_(2, indices, 1)
        return adjacency_matrices

    def process_graph(self, adjs):
        B, N, _ = adjs.size()
        degrees = torch.sum(adjs, dim=2)
        D = torch.diag_embed(degrees)
        D_sqrt_inv = torch.sqrt(torch.reciprocal(D))
        D_sqrt_inv[torch.isinf(D_sqrt_inv)] = 0.0

        normalized_adj_matrices = torch.eye(N, device=D.device).unsqueeze(0).repeat(B, 1, 1) - torch.bmm(torch.bmm(D_sqrt_inv, adjs), D_sqrt_inv.transpose(1, 2))

        return normalized_adj_matrices

    def address_overfitting_graph(self, adjs):
        B, N, _ = adjs.size()
        indentity = torch.eye(N, device=adjs.device).unsqueeze(0).repeat(B, 1, 1)
        adjs_hat = adjs + indentity
        D_hat = torch.diag_embed(torch.sum(adjs_hat, dim=2))
        D_hat_sqrt_inv = torch.sqrt(torch.reciprocal(D_hat))
        D_hat_sqrt_inv[torch.isinf(D_hat_sqrt_inv)] = 0.0
        new_matrix = torch.bmm(torch.bmm(D_hat_sqrt_inv, adjs_hat), D_hat_sqrt_inv)

        return new_matrix

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

class GCNPolynomial(nn.Module):
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
    
    def get_adj(self, x):
        B, N, _ = x.size()
        distances = torch.cdist(x, x, p=2) # (B, N, N)
        _, indices = torch.topk(distances, k=self.k, dim=-1, largest=False)
        adjacency_matrices = torch.zeros(B, N, N, device=x.device)
        adjacency_matrices.scatter_(2, indices, 1)
        return adjacency_matrices

    def process_graph(self, adjs):
        B, N, _ = adjs.size()
        degrees = torch.sum(adjs, dim=2)
        D = torch.diag_embed(degrees)
        D_sqrt_inv = torch.sqrt(torch.reciprocal(D))
        D_sqrt_inv[torch.isinf(D_sqrt_inv)] = 0.0

        normalized_adj_matrices = torch.eye(N, device=D.device).unsqueeze(0).repeat(B, 1, 1) - torch.bmm(torch.bmm(D_sqrt_inv, adjs), D_sqrt_inv.transpose(1, 2))

        return normalized_adj_matrices

    def address_overfitting_graph(self, adjs):
        B, N, _ = adjs.size()
        indentity = torch.eye(N, device=adjs.device).unsqueeze(0).repeat(B, 1, 1)
        adjs_hat = adjs + indentity
        D_hat = torch.diag_embed(torch.sum(adjs_hat, dim=2))
        D_hat_sqrt_inv = torch.sqrt(torch.reciprocal(D_hat))
        D_hat_sqrt_inv[torch.isinf(D_hat_sqrt_inv)] = 0.0
        new_matrix = torch.bmm(torch.bmm(D_hat_sqrt_inv, adjs_hat), D_hat_sqrt_inv)

        return new_matrix
            
            
        
        

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

class GCN(nn.Module):
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
        """
        input:
        x:[batch_size, N, 3]
        """
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
    
    # def cal_batch_adj(self, x):
    #     # ipdb.set_trace()
    #     device = x.device
    #     x = x.cpu().numpy()
    #     x = x.reshape(-1, 3)
    #     knn = NearestNeighbors(n_neighbors=self.k)
    #     knn.fit(x)
    #     adj = knn.kneighbors_graph(x, mode='connectivity').toarray()
    #     adj = torch.from_numpy(adj).to(device)
    #     # adj = adj.unsqueeze(0)
    #     return adj

    # def cal_adj(self, x):
    #     # ipdb.set_trace()
    #     adjs = []
    #     B = x.size(0)
    #     for i in range(B):
    #         batch_points = x[i]
    #         batch_adj = self.cal_batch_adj(batch_points)
    #         adjs.append(batch_adj)
    #     # ipdb.set_trace()
    #     adjs = torch.stack(adjs, dim=0)

    #     return adjs
    
    def get_adj(self, x):
        B, N, _ = x.size()
        distances = torch.cdist(x, x, p=2) # (B, N, N)
        _, indices = torch.topk(distances, k=self.k, dim=-1, largest=False)
        adjacency_matrices = torch.zeros(B, N, N, device=x.device)
        adjacency_matrices.scatter_(2, indices, 1)
        return adjacency_matrices


    
    def process_graph(self, adjs):
        B, N, _ = adjs.size()
        degrees = torch.sum(adjs, dim=2)
        D = torch.diag_embed(degrees)
        D_sqrt_inv = torch.sqrt(torch.reciprocal(D))
        D_sqrt_inv[torch.isinf(D_sqrt_inv)] = 0.0

        normalized_adj_matrices = torch.eye(N, device=D.device).unsqueeze(0).repeat(B, 1, 1) - torch.bmm(torch.bmm(D_sqrt_inv, adjs), D_sqrt_inv.transpose(1, 2))

        return normalized_adj_matrices

    def address_overfitting_graph(self, adjs):
        B, N, _ = adjs.size()
        indentity = torch.eye(N, device=adjs.device).unsqueeze(0).repeat(B, 1, 1)
        adjs_hat = adjs + indentity
        D_hat = torch.diag_embed(torch.sum(adjs_hat, dim=2))
        D_hat_sqrt_inv = torch.sqrt(torch.reciprocal(D_hat))
        D_hat_sqrt_inv[torch.isinf(D_hat_sqrt_inv)] = 0.0
        new_matrix = torch.bmm(torch.bmm(D_hat_sqrt_inv, adjs_hat), D_hat_sqrt_inv)

        return new_matrix


