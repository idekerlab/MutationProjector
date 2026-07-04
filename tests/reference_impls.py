"""
Pre-refactor ("old") reference implementations, kept only for equivalence testing against
the refactored src/ code. These are frozen copies of the original algorithms (per-gene Python
loops, PyG Data/DataLoader construction, pandas GATidx lookup, pd.cut binning) -- do not "fix"
them to match src/; their whole purpose is to be the un-refactored baseline.
"""
import copy
from collections import defaultdict
from itertools import product

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch_geometric
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATv2Conv


class OldGATv2_onehop(nn.Module):
    def __init__(self, num_node_features, num_heads, self_loop, GAT_dropout=0):
        super().__init__()
        self.num_node_features = num_node_features
        self.conv1 = GATv2Conv(self.num_node_features, self.num_node_features, heads=num_heads, concat=False, add_self_loops=self_loop, dropout=GAT_dropout)

    def forward(self, x, network_edge, batch_size, return_attention_weights):
        if return_attention_weights == False:
            x = self.conv1(x, network_edge)
            x = x.view(batch_size, -1, self.num_node_features)
            return x
        else:
            x, x2 = self.conv1(x, network_edge, return_attention_weights=return_attention_weights)
            x = x.view(batch_size, -1, self.num_node_features)
            edge_index, attention_weights = x2
            return x, edge_index, attention_weights


class OldGATv2block(nn.Module):
    def __init__(self, num_genes, num_features, network_edges, num_GATblock, num_heads, dropout_p, cuda_device, d_ff=100, self_loop=True):
        super().__init__()
        self.num_genes = num_genes
        self.num_features = num_features
        self.network_edges = network_edges
        self.num_GATblock = num_GATblock
        self.num_heads = num_heads
        self.dropout_p = dropout_p
        self.cuda_device = cuda_device
        self.d_ff = d_ff
        self.self_loop = self_loop

        GAT_layers = []
        self.GATidx = defaultdict(list)
        count = 0
        for i in range(self.num_GATblock):
            for j in range(len(self.network_edges)):
                GAT_layers.append(OldGATv2_onehop(self.num_features, self.num_heads, self.self_loop, GAT_dropout=self.dropout_p))
                self.GATidx['i'].append(i)
                self.GATidx['j'].append(j)
                self.GATidx['idx'].append(count)
                count += 1
        self.GAT_layers = nn.Sequential(*GAT_layers)
        self.GATidx = pd.DataFrame(self.GATidx)

        self.linear_layers = nn.ModuleList([nn.Linear(num_features * len(self.network_edges), num_features) for _ in range(self.num_genes)])
        self.FF_layer1 = nn.ModuleList([nn.Linear(self.num_features, self.d_ff) for _ in range(self.num_genes)])
        self.FF_layer2 = nn.ModuleList([nn.Linear(self.d_ff, self.num_features) for _ in range(self.num_genes)])
        self.dropout = nn.Dropout(p=self.dropout_p)
        self.layer_norm1 = nn.ModuleList([nn.LayerNorm(self.num_features) for _ in range(self.num_GATblock)])
        self.layer_norm2 = nn.ModuleList([nn.LayerNorm(self.num_features) for _ in range(self.num_GATblock)])

    def forward(self, X_input, return_attention_weights):
        X_original = X_input.clone().cuda(self.cuda_device)
        batch_size = X_input.shape[0]
        out_edges, out_att_weights = {}, {}

        for i in range(self.num_GATblock):
            for j in range(len(self.network_edges)):
                if i == 0:
                    X_train = X_original.clone()
                else:
                    X_train = X.clone()
                data_list = [Data(x=X_train[k], edge_index=self.network_edges[j]) for k in range(X_train.shape[0])]
                loader = torch_geometric.loader.DataLoader(data_list, batch_size=batch_size)
                GATidx = self.GATidx.loc[self.GATidx['i']==i,:].loc[self.GATidx['j']==j,:]['idx'].tolist()[0]
                GATlayer = self.GAT_layers[GATidx].cuda(self.cuda_device)

                for batch in loader:
                    if return_attention_weights == False:
                        x = GATlayer(batch.x.cuda(self.cuda_device), batch.edge_index.cuda(self.cuda_device), X_train.shape[0], False)
                    else:
                        x, edge_index, attention_weights = GATlayer(batch.x.cuda(self.cuda_device), batch.edge_index.cuda(self.cuda_device), X_train.shape[0], return_attention_weights)
                        out_edges['%s_%s'%(i, j)] = edge_index
                        out_att_weights['%s_%s'%(i, j)] = attention_weights

                if j == 0:
                    x_cat = x
                else:
                    x_cat = torch.cat((x_cat, x), dim=2)

            X_gat = []
            for gene_i in range(self.num_genes):
                out_gene = self.linear_layers[gene_i](x_cat[:,gene_i,:])
                X_gat.append(out_gene)
            X_gat = torch.stack(X_gat, dim=1)
            X = self.layer_norm1[i](X_train + self.dropout(X_gat))

            X_ff = []
            for gene_i in range(self.num_genes):
                out_gene = self.FF_layer1[gene_i](X[:,gene_i,:])
                out_gene = torch.relu(out_gene)
                out_gene = self.FF_layer2[gene_i](out_gene)
                X_ff.append(out_gene)
            X_ff = torch.stack(X_ff, dim=1)
            X = self.layer_norm2[i](X + self.dropout(X_ff))

        if return_attention_weights == False:
            return X
        else:
            return X, out_edges, out_att_weights


class OldTokenizeSpecialTokens(nn.Module):
    def __init__(self, num_features, num_bins, cuda_device):
        super().__init__()
        self.num_features = num_features
        self.num_bins = num_bins
        self.cuda_device = cuda_device
        self.token_emb = nn.Embedding(self.num_bins, self.num_features).cuda(self.cuda_device)

    def forward(self, values, apply_padding=False):
        if type(values) == list or type(values) == np.ndarray:
            temp = pd.DataFrame({'value': values})
        else:
            temp = pd.DataFrame({'value': values.detach().cpu().numpy()})
        if apply_padding == False:
            temp['binned'] = pd.cut(temp['value'], bins=self.num_bins, labels=np.arange(self.num_bins))
            out_emb = self.token_emb(torch.tensor(temp['binned'].tolist()).cuda(self.cuda_device))
        else:
            padding_emb = nn.Embedding(1, self.num_features, padding_idx=0).cuda(self.cuda_device)
            temp['binned'] = [0]*temp.shape[0]
            out_emb = padding_emb(torch.tensor(temp['binned'].tolist()).cuda(self.cuda_device))
        return out_emb


def old_convert_mutations(X):
    out = []
    alt_combo = list(product([0, 1], repeat=X.shape[-1]))
    vocab = {}
    for value, key in enumerate(alt_combo):
        vocab[key] = value
    vocab['masked'] = value + 1
    for i in range(X.shape[0]):
        temp = [vocab[tuple(X[i][j].numpy())] for j in range(X.shape[1])]
        out.append(temp)
    return torch.tensor(np.array(out))


class OldTokenizer(nn.Module):
    def __init__(self, num_genes, num_features, cuda_device, num_gene_features=3, input_genes=[], cls_token=False, mask_nonzeros=False, gene_feature_index_to_mask=0):
        super().__init__()
        self.num_genes = num_genes
        self.num_features = num_features
        self.num_gene_features = num_gene_features
        self.cuda_device = cuda_device
        self.cls_token = cls_token
        self.mask_nonzeros = mask_nonzeros
        self.gene_feature_index_to_mask = gene_feature_index_to_mask
        self.alt_combo = list(product([0, 1], repeat=self.num_gene_features))
        self.vocab = {}
        for value, key in enumerate(self.alt_combo):
            self.vocab[key] = value
        self.vocab['masked'] = value + 1
        self.gene_embedding = nn.Embedding(self.num_genes, self.num_features).cuda(self.cuda_device)
        self.mut_embedding = nn.Embedding(len(self.vocab.keys()), self.num_features).cuda(self.cuda_device)

    def return_gene_embedding(self, num_samples):
        N_gene_emb = self.num_genes
        out = self.gene_embedding(torch.tensor(np.arange(N_gene_emb)).cuda(self.cuda_device)).unsqueeze(0).repeat(num_samples, 1, 1).cuda(self.cuda_device)
        return out

    def return_mut_embedding(self, X_converted, mask_percentage, test_geneset, positions_to_mask=None):
        N_gene_emb = self.num_genes
        out = torch.zeros(N_gene_emb, self.num_features, requires_grad=True).cuda(self.cuda_device)
        out = out.unsqueeze(0).repeat(X_converted.shape[0], 1, 1)
        if positions_to_mask is None:
            positions_to_mask = []
            if test_geneset == False:
                num_to_mask = self.num_genes * mask_percentage // 100
            else:
                positions_to_mask = test_geneset
                num_to_mask = len(positions_to_mask)
        else:
            num_to_mask = len(positions_to_mask)
        if num_to_mask > 0:
            masked_emb = self.mut_embedding(torch.tensor(self.vocab['masked']).cuda(self.cuda_device))
            out[:,positions_to_mask,:] = masked_emb.repeat(X_converted.shape[0], 1, 1)
        for value, key in enumerate(self.alt_combo):
            i_, j_ = torch.where(X_converted == torch.tensor(value))
            mut_emb = self.mut_embedding(torch.tensor(value).cuda(self.cuda_device))
            for i, j in zip(i_, j_):
                if not j.item() in positions_to_mask:
                    out[i][j] = mut_emb
        return out, positions_to_mask

    def forward(self, X_converted, mask_percentage, test_geneset):
        gene_emb = self.return_gene_embedding(X_converted.shape[0])
        mut_emb, positions_to_mask = self.return_mut_embedding(X_converted, mask_percentage, test_geneset)
        out = gene_emb + mut_emb
        return out, positions_to_mask


class OldMutationProjector(nn.Module):
    def __init__(self, num_genes, num_features, network_edges, num_GATblock, num_heads, dropout_p, cuda_device, output_sizes,
                 mask_percentage, input_genes=[], d_ff=100,
                 use_representative_embedding=True, ssl_task_index=-1,
                 use_special_token=False, num_special_tokens=0, num_bins=[],
                 use_pooling=True):
        super().__init__()
        self.num_genes = num_genes
        self.num_features = num_features
        self.network_edges = network_edges
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
        self.num_networks = len(self.network_edges)

        if self.use_special_token == False:
            self.GATblock = OldGATv2block(self.num_genes, self.num_features, self.network_edges, self.num_GATblock, self.num_heads, self.dropout_p, self.cuda_device, self.d_ff, self_loop=False)
        else:
            assert self.num_special_tokens >= 0
            assert len(self.num_bins) == self.num_special_tokens
            self.special_tokenizer = nn.ModuleList(
                [OldTokenizeSpecialTokens(self.num_features, self.num_bins[i], self.cuda_device) for i in range(self.num_special_tokens)])
            new_network_edges = copy.deepcopy(network_edges)
            self.num_nodes = self.num_genes
            special_token_edges = [[], []]
            for s_idx in range(self.num_special_tokens):
                special_token_idx = self.num_genes + s_idx
                for node_idx in range(self.num_nodes):
                    special_token_edges[0].append(node_idx); special_token_edges[1].append(special_token_idx)
                    special_token_edges[0].append(special_token_idx); special_token_edges[1].append(node_idx)
                self.num_nodes = self.num_nodes + 1
            special_token_edges = torch.tensor(special_token_edges)
            for network_idx, network in enumerate(new_network_edges):
                new_network_edges[network_idx] = torch.cat((network, special_token_edges), dim=1)
            self.GATblock = OldGATv2block(self.num_nodes, self.num_features, new_network_edges, self.num_GATblock, self.num_heads, self.dropout_p, self.cuda_device, self.d_ff, self_loop=False)

        self.tokenizer = OldTokenizer(self.num_genes, self.num_features, self.cuda_device, input_genes=self.input_genes)
        self.dropout = nn.Dropout(p=self.dropout_p)
        self.gene_emb_size = self.num_genes * self.num_features

        self.final_linear1 = nn.ModuleList()
        if (self.use_special_token == False) and (self.use_representative_embedding == False):
            for i in range(len(self.output_sizes)):
                if i == self.ssl_task_index:
                    prot2gene_layer = nn.ModuleList([nn.Linear(self.num_features, self.output_sizes[i]) for _ in range(self.num_genes)])
                    self.final_linear1.append(prot2gene_layer)
                else:
                    self.final_linear1.append(nn.Linear(self.gene_emb_size, self.output_sizes[i]))
        else:
            special_token_emb_size, rep_emb_size = 0, 0
            if self.use_representative_embedding == True:
                if self.use_pooling == True:
                    rep_emb_size = self.num_features*3
                else:
                    rep_emb_size = self.num_features*self.num_special_tokens
                    self.FFNN = nn.Sequential(
                        nn.Linear(self.num_genes*self.num_features, self.num_features*self.num_special_tokens),
                        nn.LayerNorm(self.num_features*self.num_special_tokens),
                        nn.ReLU())
            if self.use_special_token == True:
                special_token_emb_size = self.num_features*self.num_special_tokens
            self.Layer_norm = nn.LayerNorm(self.num_features)
            for i in range(len(self.output_sizes)):
                if i == self.ssl_task_index:
                    prot2gene_layer = nn.ModuleList([nn.Linear(self.num_features, self.output_sizes[i]) for _ in range(self.num_genes)])
                    self.final_linear1.append(prot2gene_layer)
                else:
                    self.final_linear1.append(nn.Linear(self.num_features, self.output_sizes[i]))
                    self.concat_FF_layer = nn.Sequential(
                        nn.Linear(special_token_emb_size+rep_emb_size, self.num_features),
                        nn.LayerNorm(self.num_features),
                        nn.ReLU())

    def forward(self, X, X_special_tokens=[], test_geneset=False, return_attention_weights=False, apply_paddings=False):
        attention_weights = torch.Tensor([])
        batch_size = X.shape[0]
        X, masked_positions = self.tokenizer(X, self.mask_percentage, test_geneset)
        if self.use_special_token == True:
            assert X_special_tokens.shape[1] == self.num_special_tokens
            for s_idx in range(self.num_special_tokens):
                if apply_paddings == False:
                    X_add = self.special_tokenizer[s_idx](X_special_tokens[:,s_idx].cuda(self.cuda_device)).cuda(self.cuda_device)
                else:
                    apply_padding = s_idx in apply_paddings
                    X_add = self.special_tokenizer[s_idx](X_special_tokens[:,s_idx].cuda(self.cuda_device), apply_padding=apply_padding).cuda(self.cuda_device)
                X_add = torch.unsqueeze(X_add, dim=1)
                X = torch.cat((X, X_add), dim=1)

        if return_attention_weights == False:
            X = self.GATblock(X, False)
        else:
            X, edge_indices, attention_weights = self.GATblock(X, True)
        gene_emb = X.clone()
        gene_emb = gene_emb[:, :self.num_genes]

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

        rep_emb = torch.tensor([]).cuda(self.cuda_device)
        if self.use_representative_embedding == True:
            if self.use_pooling == True:
                mean_pool = torch.mean(gene_emb, dim=1)
                max_pool, _ = torch.max(gene_emb, dim=1)
                min_pool, _ = torch.min(gene_emb, dim=1)
                mean_pool, max_pool, min_pool = self.Layer_norm(mean_pool), self.Layer_norm(max_pool), self.Layer_norm(min_pool)
                rep_emb = torch.cat((mean_pool, max_pool, min_pool), dim=1).cuda(self.cuda_device)
            else:
                rep_emb = self.FFNN(gene_emb.reshape(gene_emb.shape[0], -1)).cuda(self.cuda_device)

        output1 = []
        for task_layer_index, task_layer in enumerate(self.final_linear1):
            if task_layer_index == self.ssl_task_index:
                masked_gene_pred = []
                for gene_idx in range(self.num_genes):
                    gene_emb2 = X.clone()
                    gene_emb2 = gene_emb2[:, :self.num_genes]
                    protein_emb = gene_emb2[:, gene_idx, :]
                    transformed_emb = self.final_linear1[task_layer_index][gene_idx](protein_emb).cuda(self.cuda_device)
                    masked_gene_pred.append(transformed_emb)
                masked_gene_pred = torch.stack(masked_gene_pred, dim=1)
                output1.append(masked_gene_pred)
            else:
                if (self.use_special_token == False) and (self.use_representative_embedding == False):
                    output1.append(task_layer(self.dropout(gene_emb.reshape(gene_emb.shape[0], -1)).cuda(self.cuda_device)))
                else:
                    if self.use_special_token == True:
                        out_concat_layer = self.concat_FF_layer(torch.cat((special_token_emb.reshape(special_token_emb.shape[0], -1), rep_emb), dim=1)).cuda(self.cuda_device)
                    else:
                        out_concat_layer = self.concat_FF_layer(rep_emb).cuda(self.cuda_device)
                    output1.append(task_layer(self.dropout(out_concat_layer).cuda(self.cuda_device)))

        if return_attention_weights == False:
            return output1, masked_positions, gene_emb
        else:
            return output1, masked_positions, attention_weights, edge_indices, (gene_emb, cov_emb, rep_emb, out_concat_layer)
