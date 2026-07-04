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
import argparse

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
from load_model import adapt_legacy_state_dict

def gen_embedding():
    #############################################
    ## User inputs
    #############################################
    parser = argparse.ArgumentParser(description='Generate MutationProjector embeddings')
    # arguments for generating embeddings
    parser.add_argument('-dataset', help='name of the dataset', type=str, default='na')
    parser.add_argument('-dataset_type', help='dataset type ("train_dataset" or "eval_dataset")', type=str, default='na')
    # args
    args = parser.parse_args()
    
    #############################################
    ## Generate embeddings
    #############################################
    print(f'Generating embeddings for "{args.dataset}", {time.ctime()}')
    embed_from_pretrained(pretrained_model='pretrained_model.pth', dataset=args.dataset, dataset_type=args.dataset_type)
    print(f'Done, results available at "../prediction_results/{args.dataset}", {time.ctime()}')
    
    


def embed_from_pretrained(pretrained_model, dataset,
                          dataset_type='train_dataset',
                          geneset='MSKIMPACT468', 
                          networks='GRN;E3;phosphorylation;physical_ppi;genetic_interaction;DDRAM;STRING;PCNET', 
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
                          weight_decay = 0.0001,
                          path_dataset=None,
                         ):
    
    # load genes
    gset = geneset
    input_genes = load_genes(gset=gset)

    #####################################
    # load data
    #####################################
    ## fi_dir
    fi_dir = Path().resolve().parent
    if path_dataset == None:
        PATH_DATA = f'{fi_dir}/data/downstream_data/{dataset_type}/{dataset}'
    else:
        PATH_DATA = path_dataset
    if os.path.exists(PATH_DATA) == False:
        print(f"Path {PATH_DATA} not found")
        
    ## genomic data
    gData, sData, pData = {}, {}, {}
    # load data
    # genomic
    mdf = pd.read_csv(f'{PATH_DATA}/mut.txt', sep='\t')
    cna = pd.read_csv(f'{PATH_DATA}/cna.txt', sep='\t')
    cnd = pd.read_csv(f'{PATH_DATA}/cnd.txt', sep='\t')
    merged = merge_data(mdf, cna, cnd, use_cancer_types=False)
    # sData
    tmp_sData = pd.read_csv(f'{PATH_DATA}/covariates.txt', sep='\t')
    # gData, sData, pData
    gData[dataset] = merged[0] 
    sData[dataset] = tmp_sData
    #####################################

    
    

    
    #####################################
    # load network
    #####################################
    networks = networks.split(';')
    network_edges = []
    for n_idx in range(len(networks)):
        network = networks[n_idx]
        edges = load_network().return_edges(network)
        network_edges.append(edges)
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
    padding_info = {'IMvigor210':list(np.arange(1, 9)),
                'mel_dfci_2019':[1],
                'mixed_allen_2018':[1]}
    if len(padding_idx) > 0:
        padding_info[dataset] = padding_idx
    # embeddings
    Gene_emb = torch.tensor([]).cuda(cuda_device)
    Rep_emb = torch.tensor([]).cuda(cuda_device)
    Cov_emb = torch.tensor([]).cuda(cuda_device)
    Final_emb = torch.tensor([]).cuda(cuda_device)
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
    output_sizes = [3, 10, 3]

    ## load model
    tmp = torch.load('%s/%s'%(dir_pretrained, model_name))
    pretrained_model = MutationProjector(num_genes, num_features, network_edges, num_GATblock, num_heads, dropout_p, cuda_device, output_sizes, mask_percentage, input_genes, dff, use_representative_embedding=use_rep, ssl_task_index=0, use_special_token=use_special_tokens, num_special_tokens=num_special_tokens, num_bins=num_bins2, use_pooling=use_pooling)
    pretrained_model.load_state_dict(adapt_legacy_state_dict(pretrained_model, tmp))

    ## padding
    apply_paddings = False
    if dataset in list(padding_info.keys()):
        apply_paddings = padding_info[dataset]        
        
    # split into minibatches 
    X_split = split_array(X2_test, batch_size)
    Xs_split = split_array(X_special, batch_size)

    # compute gene embeddings
    pretrained_model.eval()
    with torch.no_grad():
        for idx in range(len(X_split)):
            X1, X2 = X_split[idx], Xs_split[idx]
            # train_emb
            pred_risk1 = pretrained_model(X1, X2, test_geneset=False, return_attention_weights=True, apply_paddings=apply_paddings)
            output1, masked_positions, attention_weights, edge_indices, (gene_emb, cov_emb, rep_emb, out_concat_layer) = pred_risk1
            # concatenate
            Gene_emb = torch.concatenate((Gene_emb, gene_emb), dim=0)
            Rep_emb = torch.concatenate((Rep_emb, rep_emb), dim=0)
            Cov_emb = torch.concatenate((Cov_emb, cov_emb), dim=0)
            Final_emb = torch.concatenate((Final_emb, out_concat_layer), dim=0)

    # write results
    out_path = f"{fi_dir}/prediction_results/{dataset}"
    if path_dataset:
        out_path = f"{path_dataset}/prediction_results"
        
    os.makedirs(out_path, exist_ok=True)
    torch.save(Gene_emb, f'{out_path}/gene_emb.pt')
    torch.save(Rep_emb, f'{out_path}/rep_emb.pt')
    torch.save(Cov_emb, f'{out_path}/cov_emb.pt')
    torch.save(Final_emb, f'{out_path}/final_layer_emb.pt')
    #####################################

    
if __name__ == '__main__':
    gen_embedding()