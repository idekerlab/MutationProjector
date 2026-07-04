# GATv2_onehop : one hop convolution
# GATv2block : integrate multiple networks
import numpy as np
import scipy.stats as stat
import sklearn
from sklearn.preprocessing import *
from sklearn.model_selection import *
from sklearn.metrics import *
from sklearn.ensemble import *
from sklearn.linear_model import *
from itertools import *
import os, time, sys, random, math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv

from device_utils import resolve_device


# BatchedPerGeneLinear: num_genes independent Linear(in_features, out_features) layers,
# applied to the gene dimension of a (batch, num_genes, in_features) tensor in one batched
# matmul instead of a Python loop over num_genes separate nn.Linear modules. Same per-gene
# distinct-weight semantics as nn.ModuleList([nn.Linear(in_features, out_features) for _ in range(num_genes)]).
class BatchedPerGeneLinear(nn.Module):
    def __init__(self, num_genes, in_features, out_features):
        super().__init__()
        self.num_genes = num_genes
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(num_genes, out_features, in_features))
        self.bias = nn.Parameter(torch.empty(num_genes, out_features))
        self.reset_parameters()

    def reset_parameters(self):
        for g in range(self.num_genes):
            nn.init.kaiming_uniform_(self.weight[g], a=math.sqrt(5))
        bound = 1 / math.sqrt(self.in_features) if self.in_features > 0 else 0
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        # x: (batch, num_genes, in_features) -> (batch, num_genes, out_features)
        return torch.einsum('bgi,goi->bgo', x, self.weight) + self.bias


# GATv2_onehop
class GATv2_onehop(nn.Module):
    def __init__(self, num_node_features, num_heads, self_loop, GAT_dropout=0):
        super(GATv2_onehop, self).__init__()
        self.num_node_features = num_node_features
        
        # GATv2 layer
        self.conv1 = GATv2Conv(self.num_node_features, self.num_node_features, heads=num_heads, concat=False, add_self_loops=self_loop, dropout=GAT_dropout)


    def forward(self, x, network_edge, batch_size, return_attention_weights):
        # GATv2 layer
        if return_attention_weights == False:
            x = self.conv1(x, network_edge)
            x = x.view(batch_size, -1, self.num_node_features)
            return x
        else:
            x, x2 = self.conv1(x, network_edge, return_attention_weights=return_attention_weights)
            x = x.view(batch_size, -1, self.num_node_features)
            edge_index, attention_weights = x2
            return x, edge_index, attention_weights

        
        
# GATv2block
class GATv2block(nn.Module):
    def __init__(self, num_genes, num_features, network_edges, num_GATblock, num_heads, dropout_p, cuda_device, d_ff=100, self_loop=True):
        super(GATv2block, self).__init__()
        self.num_genes = num_genes
        self.num_features = num_features
        self.network_edges = network_edges
        self.num_GATblock = num_GATblock
        self.num_heads = num_heads
        self.dropout_p = dropout_p
        self.cuda_device = resolve_device(cuda_device)
        self.d_ff = d_ff
        self.self_loop = self_loop
        
        # GATv2 layers: directly (i, j)-indexable, no separate index map needed
        self.GAT_layers = nn.ModuleList([
            nn.ModuleList([
                GATv2_onehop(self.num_features, self.num_heads, self.self_loop, GAT_dropout=self.dropout_p)
                for j in range(len(self.network_edges))
            ])
            for i in range(self.num_GATblock)
        ])

        # cache of precomputed batched (block-diagonal-offset) edge_index, keyed by (network_idx, batch_size)
        self._edge_index_cache = {}


        # Linear transformation (per-gene distinct weights, applied as one batched op)
        self.linear_layers = BatchedPerGeneLinear(self.num_genes, num_features * len(self.network_edges), num_features).to(self.cuda_device)
        self.FF_layer1 = BatchedPerGeneLinear(self.num_genes, self.num_features, self.d_ff).to(self.cuda_device)
        self.FF_layer2 = BatchedPerGeneLinear(self.num_genes, self.d_ff, self.num_features).to(self.cuda_device)

        # dropout
        self.dropout = nn.Dropout(p=self.dropout_p)
        
        # layer norm
        self.layer_norm1 = nn.ModuleList([nn.LayerNorm(self.num_features).to(self.cuda_device) for _ in range(self.num_GATblock)])
        self.layer_norm2 = nn.ModuleList([nn.LayerNorm(self.num_features).to(self.cuda_device) for _ in range(self.num_GATblock)])
        
        
    def _get_batched_edge_index(self, network_idx, batch_size):
        # Precompute (once per (network, batch_size)) the block-diagonal offset edge_index that
        # PyG's Batch.from_data_list would produce for `batch_size` copies of this static graph:
        # each copy's edge_index offset by k*num_nodes, concatenated. Network topology and node
        # count never change across forward calls, so this only needs to be built once.
        key = (network_idx, batch_size)
        if key not in self._edge_index_cache:
            edge_index = self.network_edges[network_idx].to(self.cuda_device)
            num_nodes = self.num_genes
            num_edges = edge_index.shape[1]
            offsets = (torch.arange(batch_size, device=edge_index.device) * num_nodes).repeat_interleave(num_edges)
            self._edge_index_cache[key] = edge_index.repeat(1, batch_size) + offsets.unsqueeze(0)
        return self._edge_index_cache[key]

    def forward(self, X_input, return_attention_weights):
        X_original = X_input.clone()
        X_original = X_original.to(self.cuda_device)
        batch_size = X_input.shape[0]

        # attention weights (output)
        out_edges, out_att_weights = {}, {}

        # GAT block
        for i in range(self.num_GATblock):
            # run GAT layer
            for j in range(len(self.network_edges)):
                # first GAT layer
                if i == 0:
                    X_train = X_original.clone()
                else:
                    X_train = X.clone()
                # flatten the batch of identical-topology graphs into one disjoint-block graph
                # (equivalent to PyG's Batch.from_data_list, without rebuilding Data/DataLoader every call)
                x_flat = X_train.reshape(batch_size * self.num_genes, self.num_features).to(self.cuda_device)
                batched_edge_index = self._get_batched_edge_index(j, batch_size)
                # GAT layer
                GATlayer = self.GAT_layers[i][j].to(self.cuda_device)

                # does not return attention weights
                if return_attention_weights == False:
                    x = GATlayer(x_flat, batched_edge_index, batch_size, False)

                # return attention weights
                else:
                    x, edge_index, attention_weights = GATlayer(x_flat, batched_edge_index, batch_size, return_attention_weights)
                    out_edges['%s_%s'%(i, j)] = edge_index
                    out_att_weights['%s_%s'%(i, j)] = attention_weights

                # concat
                if j == 0:
                    x_cat = x
                else:
                    x_cat = torch.cat((x_cat, x), dim=2)
                
            # linear layer for GAT outputs (one batched op instead of a per-gene Python loop)
            X_gat = self.linear_layers(x_cat)

            # Add residual connections and normalize
            X = self.layer_norm1[i](X_train + self.dropout(X_gat))

            # Feed Forward linear layer (one batched op instead of a per-gene Python loop)
            X_ff = self.FF_layer2(torch.relu(self.FF_layer1(X)))

            # Add residual connections and normalize
            X = self.layer_norm2[i](X + self.dropout(X_ff))

            
        if return_attention_weights == False:
            return X
        else:
            return X, out_edges, out_att_weights
        