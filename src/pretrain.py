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

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
import torch_geometric
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from load_geneList import *
from loss import *
from GATv2_functions import *
from nn_training_functions import *


def pretrain_MutationProjector(X_original, input_genes, y, network_edges, num_features, num_GATblock,
                               device, batch_size, max_epoch, lr, dropout_p,
                               out_name='out',
                               out_dir=None,
                               num_heads=1,
                               use_random_seed=False, random_seed=42,
                               adamw_weight_decay=1e-2,
                               test_geneset=False, mask_percentage=15,
                               save_model_every_25_epochs=True,
                               d_ff=100,
                               use_representative_embedding=True,
                               ssl_task_index=-1,
                               clip_gradient=False,
                               use_covariates=False, covariates=[], num_bins=[5,2,2],
                               use_pooling=False,
                               loss_weights=None
                    ):
    '''
    Run multi-task learning on pre-training data
    #========
    Inputs
    #========
    X_original: Original input features [(N_samples, N_genes, N_features), ...]
    input_genes: List of genes
    y: output labels. Provide in nested list. If 'ssl' is given, will select 'mask_percentage' of genes to mask and conduct self-supervised learning. 
    network_edges: must provde in list (e.g. [network 1 edges, network2 edges, ...])
    num_features: number of features per node 
    num_GATblock: number of GAT blocks 
    device: device (e.g. 'cpu', 'cuda', 'cuda:0')
    num_heads, random_seed, adamw_weight_decay: model_hyperparameters
    save_model_every_25_epochs: default (True). Will save model parameters every 25 epoch
    out_name: default ('out'). output file name
    out_dir: directory to write the output file 
    d_ff: FF dimension
    use_representative_embedding: default (True). Use representative embeddings for gene mutations
    ssl_task_index: default -1. Index of masked gene prediction task in the "y" input. 
    clip_gradient: default (False). Clip gradient
    use_covariates: default (False). use covariates (e.g. tmb, aneuploidy and/or mutational signatures)
    covariates: list of covariates. [(N_samples, N_features), ...]
    num_bins: number of bins to use for tmb / aneuploidy tokens (e.g. [5, 2, 2])
    use_pooling: default (False). use pooling layer to generate representative embedding
    loss_weights: default (None). use list to give weights to task losses
    
    #========
    Outputs
    #========
    model, loss_aggregate
    '''

    ## X
    X = convert_mutations(X_original)
    
    ## Check input datatype
    assert type(network_edges) == list, 'wrong input data type (network_edges). Provide list'
    max_idx = 0
    if len(X) >= 2:
        for data_i, data_j in combinations(range(len(X)), 2):
            assert X[data_i].shape[1] == X[data_j].shape[1], 'wrong input tokenized gene size'
            assert X_original[data_i].shape[1] == X_original[data_j].shape[1], 'wrong input gene size'
            assert X_original[data_i].shape[2] == X_original[data_j].shape[2], 'wrong input gene feature size'
            if use_covariates:
                assert covariates[data_i].shape[1] == covariates[data_j].shape[1], 'wrong covariate size'
            # check if tasks are ordered correctly across different cohorts
            min_overlapping_tasks = np.min([len(y[data_i]), len(y[data_j])])
            for tmp_i in range(min_overlapping_tasks):
                if (not 'ssl' == y[data_i][tmp_i]) and (not 'ssl' == y[data_j][tmp_i]):
                    assert np.array(y[data_i][tmp_i]).shape[1] == np.array(y[data_j][tmp_i]).shape[1], 'wrong pre-training task size'
                
    
        # cohort with the max number of tasks 
        num_tasks = [len(y[i]) for i in range(len(y))]
        max_idx = num_tasks.index(np.max(num_tasks))
            
            
    ## num genes
    num_genes = X_original[0].shape[1]
    
    
    ## output label
    output_sizes = []
    for tmp_y in y[max_idx]:
        # auxiliary tasks
        if (not str(tmp_y) == 'ssl') and (not str(tmp_y) == 'None'): # ssl: self supervised learning
            assert type(tmp_y) == torch.Tensor, 'wrong input (y) data type' 
            output_size = tmp_y.shape[1]
            
        # self supervised learning task
        elif str(tmp_y) == 'ssl':
            num_to_mask = X[max_idx].shape[1] * mask_percentage // 100
            output_size = X_original[max_idx].shape[2] 
        output_sizes.append(output_size)

        
    ## number of covariates
    num_covariates=0
    if use_covariates:
        num_covariates = covariates[max_idx].shape[1]

    
    ## create model instance
    model = MutationProjector(num_genes, num_features, network_edges, num_GATblock, num_heads, dropout_p, device, output_sizes, mask_percentage, input_genes, d_ff, use_representative_embedding, ssl_task_index, use_special_token=use_covariates, num_special_tokens=num_covariates, num_bins=num_bins, use_pooling=use_pooling)

    # random seed
    if use_random_seed == True:
        torch.manual_seed(random_seed)

    # optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=adamw_weight_decay)


    
    ##-------------------
    ## TRAIN
    train_losses = []
    
    #for epoch in tqdm(range(max_epoch)):
    for epoch in range(max_epoch):
        if (epoch < 11) or ((epoch+1)%25 == 0):
            print('%s, %s'%(epoch+1, time.ctime()))
        # enable gradient
        model.train(True)
        running_loss = 0.
        
        # train for each cohort
        for data_i in range(len(X)):
            # data
            X2, X_original2, y2 = X[data_i], X_original[data_i], y[data_i]
            if use_scovariates:
                X_covariates = covariates[data_i]

            # minibatch
            minibatch_indices = batch_index(X2.shape[0], batch_size, random_state=epoch, drop_last=True)
            minibatch_loss = 0.

            # train 
            for idx, minibatch_idx in enumerate(minibatch_indices):
                # tokenized
                inputs = X2[minibatch_idx].to(device)
                covariates_inputs = []
                if use_covariates:
                    covariates_inputs = X_covariates[minibatch_idx].to(device)


                # zero grad
                optimizer.zero_grad()

                # make predictions
                PRED, masked_positions, gene_embeddings = model(inputs, covariates_inputs, test_geneset=False)


                # labels and loss function 
                LABELS = [] 
                loss_fns = defaultdict(list)
                for y_idx, tmp_y in enumerate(y2):
                    ## self-supervised learning
                    if str(tmp_y) == 'ssl':
                        '''
                        Loss function for self supervised learning
                        '''
                        #labels
                        labels = X_original2[minibatch_idx] #[:,masked_positions]
                        LABELS.append(labels)
                        # define loss_fn here
                        for j_ in range(labels.shape[2]):
                            flattened_labels = list(labels[:,:,j_].detach().cpu().numpy().ravel())
                            if flattened_labels.count(0) * flattened_labels.count(1) > 0:
                                class_freq = torch.tensor([flattened_labels.count(0), flattened_labels.count(1)])
                                pos_weight = class_freq[0]/class_freq[1]
                                loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
                            else:
                                loss_fn = nn.BCEWithLogitsLoss()
                            loss_fns[y_idx].append(loss_fn)


                    ## Supervised learning
                    else:
                        '''
                        Loss function for supervised learning
                        '''
                        labels = tmp_y[minibatch_idx].type(torch.float).to(device)
                        LABELS.append(labels)
                        # define loss_fn
                        # binary classification task
                        if len(set(labels.detach().cpu().ravel().numpy())) <= 2:
                            for j_ in range(labels.shape[1]):
                                flattened_labels = list(labels[:,j_].detach().cpu().numpy().ravel())
                                if flattened_labels.count(0) * flattened_labels.count(1) > 0:
                                    class_freq = torch.tensor([flattened_labels.count(0), flattened_labels.count(1)])
                                    pos_weight = class_freq[0]/class_freq[1]
                                    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
                                else:
                                    loss_fn = nn.BCEWithLogitsLoss()
                                loss_fns[y_idx].append(loss_fn)

                        # regression task
                        elif len(set(labels.detach().cpu().ravel().numpy())) > 2:
                            loss_fn = nn.MSELoss()
                            loss_fns[y_idx].append(loss_fn)




                ##=================
                # loss
                loss_aggregate = 0.

                for task_i, tmp_y in enumerate(y2):
                    loss_fn = loss_fns[task_i]
                    pred = PRED[task_i]
                    labels = LABELS[task_i]
                    
                    # loss weight
                    if loss_weights == None:
                        loss_w = 1
                    else:
                        assert type(loss_weights) == list, 'loss_weights parameter should be a list'
                        loss_w = loss_weights[task_i]
                        

                    # self supervised learning task
                    if str(tmp_y) == 'ssl':
                        tmp_pred = pred.view(pred.shape[0], X_original2.shape[1], X_original2.shape[2])
                        for i in range(tmp_pred.shape[0]):
                            for j in range(tmp_pred.shape[2]):
                                loss_ = loss_fn[j](tmp_pred[:,masked_positions][i][:,j].to(device), labels[:,masked_positions][i][:,j].type(torch.float).to(device))
                                loss_ = loss_/num_to_mask # tmp_pred.shape[2]
                                loss_aggregate += loss_ * loss_w

                    # supervised learning
                    else:
                        for i in range(pred.shape[1]):
                            loss_ = loss_fn[i](pred[:,i], labels[:,i])
                            loss_ = loss_/pred.shape[1]
                            loss_aggregate += loss_ * loss_w
                    ##=================

                # loss backprop
                loss_aggregate.backward(retain_graph=True)

                # clip gradient
                if clip_gradient == True:
                    clip_grad_norm_(model.parameters(), max_norm=10.0)

                # adjust optimizer weight
                optimizer.step()
                # add loss
                minibatch_loss += loss_aggregate.item()

        
        
        
        ##=============================================================================
        # running loss
        running_loss = minibatch_loss/len(minibatch_indices)
        train_losses.append(running_loss)
        if (epoch < 11) or ((epoch+1)%25 == 0):
            print(running_loss)
        
        # save model 
        if save_model_every_25_epochs == True:
            assert out_dir == None, 'provide "out_dir" parameter'
            if epoch == 0 or (epoch + 1) % 25 == 0:
                torch.save(model.state_dict(), '%s/%s_%s.pth'%(out_dir, out_name, epoch+1))
                
                
    return model, train_losses


