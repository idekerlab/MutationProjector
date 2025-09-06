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
from tqdm import tqdm
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

## load tools
from load_geneList import *
from nn_training_functions import *
from import_network import *
from nn_training_functions import *
from MutationProjector_nn import *
from GATv2_attention_weights import *


def embed_from_pretrained(pretrained_model, dataset,
                          dataset_type='train_dataset',
                          geneset='MSKIMPACT468', 
                          networks='GRN;E3;phosphorylation;physical_ppi;SL;DDRAM;STRING;PCNET', 
                          padding_idx=[],
                          split_train_data=0,
                          num_features=10,
                          num_GATblock=2,
                          dff=10,
                          use_rep=1,
                          use_pooling=0,
                          use_gradclip=0,
                          use_special_tokens=1,
                          num_bins=5,
                          epoch=100,
                          cuda_device=0,
                          lr=0.001,
                          dropout_p=0.1,
                          num_heads=1,
                          mask_percentage=0,
                          batch_size = 64,
                          weight_decay = 0.0001
                         ):
    print(f'Generating embeddings from a pretrained model, {time.ctime()}')
    
    # load genes
    gset = geneset 
    input_genes = load_genes(gset=gset)
    print(f'Geneset name: {gset}\nNumber of genes: {len(input_genes)}')

    #####################################
    # load data
    #####################################
    ## fi_dir
    fi_dir = Path().resolve().parent
    ## genomic data
    gData, sData, pData = {}, {}, {}
    # load data
    # genomic
    mdf = pd.read_csv(f'{fi_dir}/data/downstream_data/{dataset_type}/{dataset}/mut.txt', sep='\t')
    cna = pd.read_csv(f'{fi_dir}/data/downstream_data/{dataset_type}/{dataset}/cna.txt', sep='\t')
    cnd = pd.read_csv(f'{fi_dir}/data/downstream_data{dataset_type}/{dataset}/cnd.txt', sep='\t')
    merged = merge_data(mdf, cna, cnd, use_cancer_types=False)
    # sData
    tmp_sData = pd.read_csv(f'{fi_dir}/data/downstream_data/{dataset_type}/{dataset}/covariates.txt', sep='\t')
    # gData, sData, pData
    gData[dataset] = merged[0] 
    sData[dataset] = tmp_sData
    print(dataset, gData[dataset].shape)
    #####################################

    
    

    
    #####################################
    # load network
    #####################################
    networks = networks.split(';')
    network_edges = []
    print(f'load networks, {time.ctime()}')
    for n_idx in tqdm(range(len(networks))):
        network = networks[n_idx]
        edges = load_network().return_edges(network, input_genes)
        edges = torch.from_numpy(edges).type(torch.long).T
        network_edges.append(edges)
    print(f'done loading networks, {time.ctime()}')    
    #####################################



    
    
    #####################################
    # MutationProjector
    #####################################
    model_name = pretrained_model #'pretrained_model.pth'
    dir_pretrained = f'{fi_dir}/pretrained_model'
    split_train_data, num_features, num_GATblock, dff, use_rep, use_pooling, use_gradclip, use_special_tokens, num_bins, epoch = [int(val) for val in [split_train_data, num_features, num_GATblock, dff, use_rep, use_pooling, use_gradclip, use_special_tokens, num_bins, epoch]]
    #####################################

    
    


    #####################################
    # attention weights
    #####################################
    def split_array(input_list, batch_size):
        # Calculate the approximate size for the first lists
        n = len(input_list)
        num_batch = n // batch_size
        # Create the first lists
        lists = [input_list[i * batch_size:(i + 1) * batch_size] for i in range(num_batch-1)]
        # Append the remaining elements to the last list
        lists.append(input_list[(num_batch-1) * batch_size:])
        return lists

    ## Generate embeddings
    # padding
    padding_info = {}
    if len(padding_idx) > 0:
        padding_info[dataset] = padding_idx
    # embeddings
    Gene_emb = torch.tensor([]).cuda(cuda_device)
    Rep_emb = torch.tensor([]).cuda(cuda_device)
    Cov_emb = torch.tensor([]).cuda(cuda_device)
    # X2_test
    X_test = gData[dataset]
    X_special = torch.tensor(sData[dataset].set_index('sample').values)
    X2_test = convert_mutations(X_test)
    # num_test_samples, num_genes, num_input_features
    num_test_samples, num_genes, num_input_features = X_test.shape
    num_special_tokens = X_special.shape[1]
    # num_bins
    num_bins2 = np.append([num_bins,num_bins], [2]*7)
    # output sizes
    output_sizes = [num_input_features, num_features, 3]

    ## load model
    tmp = torch.load('%s/%s'%(dir_pretrained, model_name))
    pretrained_model = MutationProjector(num_genes, num_features, network_edges, num_GATblock, num_heads, dropout_p, cuda_device, output_sizes, mask_percentage, input_genes, dff, use_representative_embedding=use_rep, ssl_task_index=0, use_special_token=use_special_tokens, num_special_tokens=num_special_tokens, num_bins=num_bins2, use_pooling=use_pooling)
    pretrained_model.load_state_dict(tmp)

    ## padding
    apply_paddings = False
    if dataset in list(padding_info.keys()):
        apply_paddings = padding_info[dataset]        
        
    # split into minibatches 
    X_split = split_array(X2_test, 64)
    Xs_split = split_array(X_special, 64)

    # compute gene embeddings
    print(f'generating embeddings, {time.ctime()}')
    pretrained_model.eval()
    with torch.no_grad():
        for idx in tqdm(range(len(X_split))):
            X1, X2 = X_split[idx], Xs_split[idx]
            # train_emb
            pred_risk1 = pretrained_model(X1, X2, test_geneset=False, return_attention_weights=True, apply_paddings=apply_paddings)
            output1, masked_positions, attention_weights, edge_indices, (gene_emb, cov_emb, rep_emb, out_concat_layer) = pred_risk1
            # concatenate
            Gene_emb = torch.concatenate((Gene_emb, gene_emb), dim=0)
            Rep_emb = torch.concatenate((Rep_emb, rep_emb), dim=0)
            Cov_emb = torch.concatenate((Cov_emb, cov_emb), dim=0)

    # write results
    torch.save(Gene_emb, f'{fi_dir}/data/downstream_data/{dataset_type}/{dataset}/gene_emb.pt')
    torch.save(Rep_emb, f'{fi_dir}/data/downstream_data/{dataset_type}/{dataset}/rep_emb.pt')
    torch.save(Cov_emb, f'{fi_dir}/data/downstream_data/{dataset_type}/{dataset}/cov_emb.pt')
    print(f'done generating embeddings, {time.ctime()}')
    #####################################