import pandas as pd
import numpy as np
from collections import defaultdict
import scipy.stats as stat
from itertools import *
import os, time, sys, random
from tqdm import tqdm
from sklearn.preprocessing import *
import sklearn
from sklearn.preprocessing import *
from sklearn.model_selection import *
from sklearn.metrics import *
from sklearn.ensemble import *
from sklearn.linear_model import *
from sklearn.decomposition import *
from sklearn.model_selection import *
import torch

def transfer_learn(downstream_train, downstream_eval, out_name=None, max_depth=10, n_estimators=100, random_state=42):
    ####################################################
    # load data
    ####################################################
    # fi_dir
    fi_dir = Path().resolve().parent
    # test if files exist
    required_files = os.listdir(f'{fi_dir}/data/downstream_data/{dataset}')
    assert len(set(['cov_emb.pt', 'rep_emb.pt', 'outcomes.txt']) & set(required_files)) == 3, "Missing one or more of the following files (cov_emb.pt, rep_emb.pt, outcomes.txt)"
    
    # inputs
    # representative gene embedding
    rep_emb1 = torch.load(f'{fi_dir}/data/downstream_data/train_dataset/{downstream_train}/cov_emb.pt').detach().cpu()
    rep_emb2 = torch.load(f'{fi_dir}/data/downstream_data/eval_dataset/{downstream_eval}/cov_emb.pt').detach().cpu()
    # covariate embedding
    cov_emb1 = torch.load(f'{fi_dir}/data/downstream_data/train_dataset/{downstream_train}/rep_emb.pt').detach().cpu()
    cov_emb2 = torch.load(f'{fi_dir}/data/downstream_data/eval_dataset/{downstream_eval}/rep_emb.pt').detach().cpu()
    # X (input data)
    X1 = torch.cat((rep_emb1, cov_emb1.reshape(cov_emb1.shape[0],-1)), dim=1)
    X2 = torch.cat((rep_emb2, cov_emb2.reshape(cov_emb2.shape[0],-1)), dim=1)
    
    # output labels
    # phenotypic outcomes
    pdf1 = pd.read_csv(f'{fi_dir}/data/downstream_data/train_dataset/{downstream_train}/outcomes.txt', sep='\t')
    pdf2 = pd.read_csv(f'{fi_dir}/data/downstream_data/train_eval/{downstream_eval}/outcomes.txt', sep='\t')
    # y (output label)
    assert 'outcomes' in pdf1.columns, f"Missing 'outcomes' column in the '{fi_dir}/data/downstream_data/train_dataset/{downstream_train}/outcomes.txt' file"
    y1 = pdf1['outcomes'].tolist()
    y2 = pdf2['outcomes'].tolist()
    

    
    ####################################################
    # train data
    ####################################################
    clf = RandomForestClassifier(random_state=random_state, n_estimators=n_estimators, max_depth=max_depth, class_weight='balanced').fit(X, y)
    
    
    ####################################################
    # output prediction results
    ####################################################
    out = pdf2.copy()
    pred_proba = clf.predict_proba(X2)[:,-1]
    out['pred_proba'] = pred_proba
    fiName = 'TransferLearning_predictions.txt'
    if out_name == None:
        out.to_csv(f'{fi_dir}/data/downstream_data/train_eval/{downstream_eval}/TransferLearning_predictions.txt', sep='\t', index=False)
    elif type(out_name) == str:
        fiName = out_name
        out.to_csv(f'{fi_dir}/data/downstream_data/train_eval/{downstream_eval}/{out_name}.txt', sep='\t', index=False)
    else:
        raise TypeError("Provide correct outcome file name for 'out_name' parameter")
    Print(f'Finished, {time.ctime()}')
    print(f"Prediction results available at : {fi_dir}/data/downstream_data/train_eval/{downstream_eval}/{fiName}")