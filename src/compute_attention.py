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
import os, time, sys, random, joblib
import matplotlib.pyplot as plt
from tqdm import tqdm
import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from load_geneList import *
from nn_training_functions import *
from GATv2_functions import *
from MutationProjector_nn import *



def split_array(input_list, batch_size):
    # Calculate the approximate size for the first lists
    n = len(input_list)
    num_batch = n // batch_size
    # Create the first lists
    lists = [input_list[i * batch_size:(i + 1) * batch_size] for i in range(num_batch-1)]
    # Append the remaining elements to the last list
    lists.append(input_list[(num_batch-1) * batch_size:])
    return lists


def compute_attention(X1, X2, edge_indices, attention_weights):
    num_samples = X1.shape[0]
    num_genes = X1.shape[1] + X2.shape[1]

    out_weights = np.full((num_samples, len(edge_indices), num_genes, num_genes), np.nan)

    for GATlayer_idx, GATlayer in enumerate(edge_indices.keys()):
        tmp_edges = edge_indices[GATlayer].detach().cpu().numpy()  # shape (2, E)
        tmp_weights = attention_weights[GATlayer].detach().cpu().numpy()  # shape (E,)

        node1 = tmp_edges[0]  # (E,)
        node2 = tmp_edges[1]  # (E,)

        sample_idx  = node1 // num_genes
        source_node = node1 - sample_idx * num_genes
        target_node = node2 - sample_idx * num_genes

        out_weights[sample_idx, GATlayer_idx, source_node, target_node] = tmp_weights.squeeze()

    return out_weights


def return_attention(alt, cov, pretrained_model, batch_size=64):

    # convert mutations
    merged2 = convert_mutations(alt)
    cov2 = torch.tensor(cov.set_index('sample').values)
    

    # split into minibatches 
    X_split = split_array(merged2, batch_size)
    Xs_split = split_array(cov2, batch_size)
    
    # pretrained model
    out = []
    pretrained_model.eval()
    with torch.no_grad():
        for idx in range(len(X_split)):
            X1, X2 = X_split[idx], Xs_split[idx]
            # run model
            pred = pretrained_model(X1, X2, test_geneset=False, return_attention_weights=True)
            output1, masked_positions, attention_weights, edge_indices, (gene_emb, cov_emb, rep_emb, out_concat_layer) = pred
            # attention
            attn = compute_attention(X1, X2, edge_indices, attention_weights)
            if len(out) == 0: out = attn
            else:
                out = np.concatenate((out, attn), axis=0)
    
    # max attention
    import warnings
    warnings.filterwarnings(
        "ignore",
        message="All-NaN slice encountered",
        category=RuntimeWarning,
    )
    max_attn = np.nanmax(out, axis=1)
    
    return out, max_attn
