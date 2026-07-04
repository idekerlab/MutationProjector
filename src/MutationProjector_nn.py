import pandas as pd
import numpy as np
from collections import defaultdict
import scipy.stats as stat
import sklearn
from sklearn.preprocessing import *
from sklearn.model_selection import *
from sklearn.metrics import *
from sklearn.ensemble import *
from sklearn.linear_model import *
from itertools import *
import os, time, sys, random
import matplotlib.pyplot as plt
from tqdm import tqdm
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_mean_pool
from load_geneList import *
from nn_training_functions import *
from loss import *
from GATv2_functions import *


class MutationProjector(nn.Module):
    def __init__(self, num_genes, num_features, network_edges, num_GATblock, num_heads, dropout_p, cuda_device, output_sizes, 
                 mask_percentage, input_genes=[], d_ff=100, 
                 use_representative_embedding=True, ssl_task_index=-1,
                 use_special_token=False, num_special_tokens=0, num_bins=[],
                 use_pooling=True

                ):
        super(MutationProjector, self).__init__()
        self.num_genes = num_genes
        self.num_features = num_features
        self.network_edges = network_edges # [[network1 edges], [network2 edges], ...]
        self.num_GATblock = num_GATblock
        self.num_heads = num_heads
        self.dropout_p = dropout_p
        self.cuda_device = cuda_device
        self.output_sizes = output_sizes
        self.mask_percentage = mask_percentage
        self.input_genes = input_genes
        self.d_ff = d_ff
        self.use_representative_embedding = use_representative_embedding
        self.ssl_task_index = ssl_task_index
        self.use_special_token = use_special_token
        self.num_special_tokens = num_special_tokens
        self.num_bins = num_bins
        self.use_pooling = use_pooling
        
        
        # num_networks
        self.num_networks = len(self.network_edges)
        

        # GATblock 
        if self.use_special_token == False:
            self.GATblock = GATv2block(self.num_genes, self.num_features, self.network_edges, self.num_GATblock, self.num_heads, self.dropout_p, self.cuda_device, self.d_ff, self_loop=False)

        elif self.use_special_token == True:
            assert self.num_special_tokens >= 0, 'mismatch'
            assert len(self.num_bins) == self.num_special_tokens, 'provide correct number of bins for special tokens'
            # special tokenizer
            self.special_tokenizer = nn.ModuleList(
                [tokenize_special_tokens(self.num_features, self.num_bins[i], self.cuda_device) for i in range(self.num_special_tokens)])
            # copy network
            new_network_edges = copy.deepcopy(network_edges)
            # update token index and number of total nodes
            self.num_nodes = self.num_genes
            # cls token edges
            special_token_edges = [[], []]
            # add edges
            for s_idx in range(self.num_special_tokens):
                # special token idx
                special_token_idx = self.num_genes + s_idx
                for node_idx in range(self.num_nodes):
                    # gene to special token
                    special_token_edges[0].append(node_idx); special_token_edges[1].append(special_token_idx)
                    # cls token to gene
                    special_token_edges[0].append(special_token_idx); special_token_edges[1].append(node_idx) 
                # add number of nodes
                self.num_nodes = self.num_nodes + 1
            # add network edges
            special_token_edges = torch.tensor(special_token_edges)
            for network_idx, network in enumerate(new_network_edges):
                new_network_edges[network_idx] = torch.cat((network, special_token_edges), dim=1)

            # GATblock
            self.GATblock = GATv2block(self.num_nodes, self.num_features, new_network_edges, self.num_GATblock, self.num_heads, self.dropout_p, self.cuda_device, self.d_ff, self_loop=False)
        
            
        # tokenizer
        self.tokenizer = tokenizer(self.num_genes, self.num_features, self.cuda_device, input_genes = self.input_genes)
            
        # dropout
        self.dropout = nn.Dropout(p=self.dropout_p)

       
        # gene_emb_size
        self.gene_emb_size = self.num_genes * self.num_features 

        
        # final linear layer (self.final_linear1)
        self.final_linear1 = nn.ModuleList()
        if (self.use_special_token == False) and (self.use_representative_embedding == False):
            for i in range(len(self.output_sizes)):
                if i == self.ssl_task_index:
                    prot2gene_layer = BatchedPerGeneLinear(self.num_genes, self.num_features, self.output_sizes[i]).cuda(self.cuda_device)
                    self.final_linear1.append(prot2gene_layer)
                else:
                    self.final_linear1.append(nn.Linear(self.gene_emb_size, self.output_sizes[i]).cuda(self.cuda_device))
                
        # use cls token and/or representative embeddings for certain tasks
        else:
            special_token_emb_size, rep_emb_size = 0, 0
            # representative embedding
            if self.use_representative_embedding == True:
                # use pooling
                if self.use_pooling == True:
                    rep_emb_size = self.num_features*3 # average / min / max pooling
                # FFNN
                else:
                    rep_emb_size = self.num_features*self.num_special_tokens                    
                    self.FFNN = nn.Sequential(
                        nn.Linear(self.num_genes*self.num_features, self.num_features*self.num_special_tokens).cuda(self.cuda_device),
                        nn.LayerNorm(self.num_features*self.num_special_tokens).cuda(self.cuda_device),
                        nn.ReLU().cuda(self.cuda_device)
                    )
            # special token embeddings
            if self.use_special_token == True:
                special_token_emb_size = self.num_features*self.num_special_tokens
            
            # layer norm
            self.Layer_norm = nn.LayerNorm(self.num_features).cuda(self.cuda_device)
            
            # final_linear1
            for i in range(len(self.output_sizes)):
                if i == self.ssl_task_index:
                    prot2gene_layer = BatchedPerGeneLinear(self.num_genes, self.num_features, self.output_sizes[i]).cuda(self.cuda_device)
                    self.final_linear1.append(prot2gene_layer)
                else:
                    self.final_linear1.append(nn.Linear(self.num_features, self.output_sizes[i]).cuda(self.cuda_device))
                    ## concat_FF_layer
                    self.concat_FF_layer = nn.Sequential(
                        nn.Linear(special_token_emb_size+rep_emb_size, self.num_features).cuda(self.cuda_device),
                        nn.LayerNorm(self.num_features).cuda(self.cuda_device),
                        nn.ReLU().cuda(self.cuda_device)
                    )
                

            
    def forward(self, X, X_special_tokens=[], test_geneset=False, return_attention_weights=False, apply_paddings=False):
        '''
        X: tokenized mutation profiles. (N_samples, N_genes)
        X_special_input: input for special tokens (N_samples, N_signatures). Give numpy or tensor
        test_geneset: False (default. does not mask genes). list (mask at input index positions)
        return_attention_weights: False (default). Returns attention weights if True
        apply_paddings: False (default). list of indices (within special tokens) to apply padding
        
        
        ##---------------
        Returns
        ##---------------
        If return_attention_weights is False:
        output1, output2, masked_positions, gene_emb
        
        Else:
        output1, output2, masked_positions, attention_weights, edge_indices, gene_emb
        
        Where,
        output1, output2: predicted values for task #1 and #2
        masked_positions: [masked gene positions]. Empty list if no masking
        gene_emb: gene embeddings
        attention_weights: attention weights
        edge_indices: edge indices 
        '''        
        attention_weights = torch.Tensor([])
        
        ## batch size
        batch_size = X.shape[0]
        
        ## tokenize
        # tokenize mutations
        X, masked_positions = self.tokenizer(X, self.mask_percentage, test_geneset)
        # tokenize special tokens
        if (self.use_special_token == True):
            assert X_special_tokens.shape[1] == self.num_special_tokens, 'provide correct X_special_tokens'
            for s_idx in range(self.num_special_tokens):
                # no padding
                if apply_paddings == False:
                    X_add = self.special_tokenizer[s_idx](X_special_tokens[:,s_idx].cuda(self.cuda_device))
                # apply padding
                else:
                    assert type(apply_paddings)==list, 'provide correct "apply_paddings" parameter'
                    apply_padding = False
                    if s_idx in apply_paddings:
                        apply_padding = True
                    X_add = self.special_tokenizer[s_idx](X_special_tokens[:,s_idx].cuda(self.cuda_device), apply_padding=apply_padding)
                # add embeddings 
                X_add = torch.unsqueeze(X_add, dim=1)
                X = torch.cat((X, X_add), dim=1)
                
        
        ## GAT blocks
        # learned gene embedding
        if return_attention_weights == False:
            X = self.GATblock(X, False)
        elif return_attention_weights == True:
            X, edge_indices, attention_weights = self.GATblock(X, True)
        gene_emb = X.clone()
        gene_emb = gene_emb[:, :self.num_genes]
        
        # apply layer norm on special tokens
        if self.use_special_token == True:
            special_token_emb = []
            cov_emb = []
            for s_idx in range(self.num_special_tokens):
                special_token = X[:, -self.num_special_tokens:][:, s_idx]
                cov_emb_ = special_token.clone()
                cov_emb.append(cov_emb_)
                special_token = self.Layer_norm(special_token)
                special_token_emb.append(special_token)
            special_token_emb = torch.stack(special_token_emb, dim=1)
            cov_emb = torch.stack(cov_emb, dim=1)
        else: 
            special_token_emb = torch.tensor([]).cuda(self.cuda_device)
            cov_emb = torch.tensor([])
        
       
        
        ## Representative embeddings
        rep_emb = torch.tensor([]).cuda(self.cuda_device)
        if self.use_representative_embedding == True:
            # pooling
            if self.use_pooling==True:
                mean_pool = torch.mean(gene_emb, dim=1)
                max_pool, _ = torch.max(gene_emb, dim=1)
                min_pool, _ = torch.min(gene_emb, dim=1)
                mean_pool, max_pool, min_pool = self.Layer_norm(mean_pool), self.Layer_norm(max_pool), self.Layer_norm(min_pool)
                rep_emb = torch.cat((mean_pool, max_pool, min_pool), dim=1)
            # FFNN
            else:
                rep_emb = self.FFNN(gene_emb.reshape(gene_emb.shape[0], -1))
        
        
        # final linear
        output1 = []
        for task_layer_index, task_layer in enumerate(self.final_linear1):
            # masked gene prediction
            if task_layer_index == self.ssl_task_index:
                # one batched op instead of a per-gene Python loop (each iteration previously
                # re-cloned the full X tensor even though only a read-only per-gene slice was used)
                gene_emb2 = X[:, :self.num_genes]
                masked_gene_pred = self.final_linear1[task_layer_index](gene_emb2)
                output1.append(masked_gene_pred)

            # other tasks
            else:
                if (self.use_special_token == False) and (self.use_representative_embedding == False):
                    output1.append(task_layer(self.dropout(gene_emb.reshape(gene_emb.shape[0], -1))))

                else:
                    if self.use_special_token == True:
                        out_concat_layer = self.concat_FF_layer(torch.cat((special_token_emb.reshape(special_token_emb.shape[0], -1), rep_emb), dim=1))
                    else:
                        out_concat_layer = self.concat_FF_layer(rep_emb)
                    output1.append(task_layer(self.dropout(out_concat_layer)))

        
        # return output
        if return_attention_weights==False:
            return output1, masked_positions, gene_emb
        else:
            return output1, masked_positions, attention_weights, edge_indices, (gene_emb, cov_emb, rep_emb, out_concat_layer)

