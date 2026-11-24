import pandas as pd
import numpy as np
from collections import defaultdict
import scipy.stats as stat
from itertools import *
import os, time, sys, random
from tqdm import tqdm
import sklearn
from sklearn.preprocessing import *
from sklearn.model_selection import *
from sklearn.metrics import *
from sklearn.ensemble import *
from sklearn.linear_model import *
from sklearn.decomposition import *
from sklearn.model_selection import *
import torch
from pathlib import Path

def transfer_learn(downstream_train,
                   downstream_eval,
                   out_name     = None,
                   max_depth    = 10,
                   n_estimators = 100,
                   random_state = 42,
                   path_train   = None,
                   path_test    = None,                   
                  ):
    ####################################################
    # load data
    ####################################################
    # fi_dir
    fi_dir = Path().resolve().parent
    PATH_TRAIN, PATH_TEST = path_train, path_test
    if path_train == None:
        PATH_TRAIN  = f'{fi_dir}/prediction_results/{downstream_train}'
    else:
        PATH_TRAIN = f"{path_train}/prediction_results"
    if path_test == None:
        PATH_TEST = f'{fi_dir}/prediction_results/{downstream_eval}'
    else:
        PATH_TEST = f"{path_test}/prediction_results"
    assert os.path.exists(PATH_TRAIN), f"Path {PATH_TRAIN} (train data) not found"
    assert os.path.exists(PATH_TEST), f"Path {PATH_TEST} (test data) not found"
    
    
    # inputs
    # representative gene embedding
    rep_emb1 = torch.load(f'{PATH_TRAIN}/rep_emb.pt', map_location='cpu').detach().cpu()
    rep_emb2 = torch.load(f'{PATH_TEST}/rep_emb.pt', map_location='cpu').detach().cpu()
    # covariate embedding
    cov_emb1 = torch.load(f'{PATH_TRAIN}/cov_emb.pt', map_location='cpu').detach().cpu()
    cov_emb2 = torch.load(f'{PATH_TEST}/cov_emb.pt', map_location='cpu').detach().cpu()
    # X (input data)
    X1 = torch.cat((rep_emb1.reshape(rep_emb1.shape[0],-1), cov_emb1.reshape(cov_emb1.shape[0],-1)), dim=1)
    X2 = torch.cat((rep_emb2.reshape(rep_emb2.shape[0],-1), cov_emb2.reshape(cov_emb2.shape[0],-1)), dim=1)
    # phenotypic outcomes
    try:
        pdf1 = pd.read_csv(f'{PATH_TRAIN}/outcomes.txt', sep='\t')
    except:
        if path_train:
            pdf1 = pd.read_csv(f"{path_train}/outcomes.txt", sep='\t')
        else:
            pdf1 = pd.read_csv(f"{fi_dir}/data/downstream_data/train_dataset/{downstream_train}/outcomes.txt", sep='\t')
    y_idx = [idx for idx in range(pdf1.shape[0]) if not pdf1['outcomes'].tolist()[idx] == 'na']
    X1 = X1[y_idx]
    y = pdf1['outcomes'].to_numpy()[y_idx].astype(int)
    # Scale data
    X1 = StandardScaler().fit_transform(X1)
    X2 = StandardScaler().fit_transform(X2)

    
    ####################################################
    # train data
    ####################################################
    clf = RandomForestClassifier(random_state=random_state, n_estimators=n_estimators, max_depth=max_depth, class_weight='balanced').fit(X1, y)
    
    
    ####################################################
    # output prediction results
    ####################################################
    try:
        pdf2 = pd.read_csv(f'{PATH_TEST}/covariates.txt', sep='\t')
    except:
        if path_test:
            pdf2 = pd.read_csv(f"{path_test}/outcomes.txt", sep='\t')
        else:
            pdf2 = pd.read_csv(f"{fi_dir}/data/downstream_data/eval_dataset/{downstream_eval}/outcomes.txt", sep='\t')
    pdf2 = pd.DataFrame(data=pdf2, columns=['sample'])
    out = pdf2.copy()
    pred_proba = clf.predict_proba(X2)[:,-1]
    out['pred_proba'] = pred_proba
    
    # fiName
    fiName = 'TransferLearning_predictions.txt'
    if out_name:
        fiName = f"{out_name}.txt"
    
    # out_dir
    out_dir = f'{fi_dir}/prediction_results/{downstream_eval}'
    if path_test:
        os.makedirs(os.path.normpath(f"{path_test}/prediction_results"), exist_ok=True)
        out_dir = os.path.normpath(f"{path_test}/prediction_results")
    
    # write results
    out.to_csv(f"{out_dir}/{fiName}", sep="\t", index=False)
    print(f'Finished, {time.ctime()}')
    print(f"Prediction results available at : {out_dir}/{fiName}")
    