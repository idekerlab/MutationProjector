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
from MutationProjector_nn import *
from GATv2_functions import *




def load_MutationProjector():
    # fi_dir
    fi_dir = Path().resolve().parent
    
    # load genes
    gset = 'MSKIMPACT468' # 'MSKIMPACT468', 'clinicalnest'
    input_genes = load_genes(gset=gset)
    
    # import network
    print(f'load networks, {time.ctime()}')
    networks = 'GRN_expanded;E3_expanded;phosphorylation_expanded;physical_ppi_expanded;genetic_interaction_expanded;DDRAM;STRING;PCNET'.split(';')
    network_edges = []
    for n_idx, network in enumerate(networks):
        edges = torch.load(f'{fi_dir}/data/networks/{network}.pt')
        network_edges.append(edges)
    print(f'finished loading networks, {time.ctime()}')

    # pretrained model
    model_name = '0-TrainTestSplit_MSKIMPACT468-genePanel_10-features_2-GATblocks_10-dff_1-useRepEmb_0-usePooling_0-gradClip_1-specialTokens_5-numBins_100.pth'
    dir_pretrained = f'{home_dir}/Projects/GPAcell/model_pretraining/pretrained_models'

    # hyperparameters
    split_train_data=0
    num_features=10
    num_GATblock=2
    dff=10
    use_rep=1
    use_pooling=0
    use_gradclip=0
    use_special_tokens=1
    num_bins=5
    epoch=100
    cuda_device=0
    lr=0.001
    dropout_p=0.1
    num_heads=1
    mask_percentage=0
    batch_size = 64
    weight_decay = 0.0001    
    cuda_device=0
    lr=0.001
    dropout_p=0.1
    num_heads=1
    mask_percentage=0
    batch_size = 64
    weight_decay = 0.0001
    # input data
    num_genes = 468
    num_input_features = 3
    num_special_tokens = 9
    # num_bins
    num_bins2 = np.append([5,5], [2]*7)
    # output sizes
    output_sizes = [3, 10, 3]


    ## load pretrained_model
    tmp = torch.load(f'{fi_dir}/pretrained_model/pretrained_model.pth')
    pretrained_model = MutationProjector(num_genes, num_features, network_edges, num_GATblock, num_heads, dropout_p, cuda_device, output_sizes, mask_percentage, input_genes, dff, use_representative_embedding=use_rep, ssl_task_index=0, use_special_token=use_special_tokens, num_special_tokens=num_special_tokens, num_bins=num_bins2, use_pooling=use_pooling)
    pretrained_model.load_state_dict(tmp)
    print('model loaded')
    
    return pretrained_model