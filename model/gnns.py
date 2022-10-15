import math
import torch
import torch.nn as nn
from torch.nn import LayerNorm, ReLU
import torch.nn.functional as F
import numpy as np
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, DeepGCNLayer
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree, softmax
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, GlobalAttention, Set2Set
import pdb

class GINConv(MessagePassing):
    def __init__(self, in_dim, out_dim, aggr = "add"):
        super(GINConv, self).__init__()
        # multi-layer perceptron
        self.mlp = torch.nn.Sequential(torch.nn.Linear(in_dim, in_dim), torch.nn.BatchNorm1d(in_dim), torch.nn.ReLU(), torch.nn.Linear(in_dim, out_dim))

        self.aggr = aggr

    def forward(self, x, edge_index):
        #add self loops in the edge space
        edge_index = add_self_loops(edge_index, num_nodes = x.size(0))

        return self.propagate(edge_index[0], x=x)

    def message(self, x_j):
        return torch.cat([x_j], dim = 1)

    def update(self, aggr_out):
        return self.mlp(aggr_out)


class GNN(torch.nn.Module):

    def __init__(self, nfeat, nhid=[128], nclass=0, out_dim=0, gnn_type='gcn', bias=True, dropout=0.5, batch_norm=False, index=-1):
        super(GNN, self).__init__()
        self.depth = len(nhid)
        self.bias = bias
        self.dropout = dropout
        self.index = index
        self.batch_norm = batch_norm
        gc = []
        for i in range(self.depth + 2):
            if i == 0:
                if gnn_type == "gin":
                    gc.append(GINConv(nfeat, nhid[0]))
                elif gnn_type == "gcn":
                    gc.append(GCNConv(nfeat, nhid[0]))
                elif gnn_type == "gat":
                    gc.append(GATConv(nfeat, nhid[0]))
                elif gnn_type == "sage":
                    gc.append(SAGEConv(nfeat, nhid[0]))
            elif i == self.depth:
                if gnn_type == "gin":
                    gc.append(GINConv(nhid[self.depth-1], nclass))    
                elif gnn_type == "gcn":
                    gc.append(GCNConv(nhid[self.depth-1], nclass))
                elif gnn_type == "gat":
                    gc.append(GATConv(nhid[self.depth-1], nclass))
                elif gnn_type == "sage":
                    gc.append(SAGEConv(nhid[self.depth-1], nclass))
            elif i == self.depth + 1:
                if gnn_type == "gin":
                    gc.append(GINConv(nhid[self.depth-1], out_dim))
                elif gnn_type == "gcn":
                    gc.append(GCNConv(nhid[self.depth-1], out_dim))
                elif gnn_type == "gat":
                    gc.append(GATConv(nhid[self.depth-1], out_dim))
                elif gnn_type == "sage":
                    gc.append(SAGEConv(nhid[self.depth-1], out_dim))
            else:
                if gnn_type == "gin":
                    gc.append(GINConv(nhid[i-1], nhid[i]))
                elif gnn_type == "gcn":
                    gc.append(GCNConv(nhid[i-1], nhid[i]))
                elif gnn_type == "gat":
                    gc.append(GATConv(nhid[i-1], nhid[i]))
                elif gnn_type == "sage":
                    gc.append(SAGEConv(nhid[i-1], nhid[i]))
        self.gc = nn.ModuleList(gc)
        if self.index == -1:
            self.bn = nn.BatchNorm1d(out_dim, 0.8)
        elif self.index == -2:
            self.bn = nn.BatchNorm1d(nclass, 0.8)

    def forward(self, x, A):
        H = x.clone()
        edges = torch.where(A > 0)
        edges = torch.cat((edges[0].view(1, -1), edges[1].view(1, -1))).long()
        H = F.dropout(H, 0.5, training=self.training)
        for i in range(0, self.depth):
            H = self.gc[i](H, edges)
            H = F.relu(H)
            H = F.dropout(H, self.dropout, training=self.training)
        if self.batch_norm == True:
            return self.bn(self.gc[self.index](H, edges))
        else:
            return self.gc[self.index](H, edges)

