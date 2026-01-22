import pandas as pd
import numpy as np
from collections import defaultdict
import scipy.stats as stat
from itertools import *
import os, time, sys, random, joblib
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

def use_transfer_learned(downstream_eval, model_name, out_name=None):
    ####################################################
    # load data
    ####################################################
    # fi_dir
    fi_dir = Path().resolve().parent
    
    # inputs
    # representative gene embedding
    rep_emb = torch.load(f'{fi_dir}/prediction_results/{downstream_eval}/rep_emb.pt').detach().cpu()
    # covariate embedding
    cov_emb = torch.load(f'{fi_dir}/prediction_results/{downstream_eval}/cov_emb.pt').detach().cpu()
    # X (input data)
    X = torch.cat((rep_emb.reshape(rep_emb.shape[0],-1), cov_emb.reshape(cov_emb.shape[0],-1)), dim=1)
    # Scale data
    X = StandardScaler().fit_transform(X)
    
    # output labels
    # phenotypic outcomes
    pdf = pd.read_csv(f'{fi_dir}/data/downstream_data/eval_dataset/{downstream_eval}/outcomes.txt', sep='\t')
    

    
    ####################################################
    # load trained model
    ####################################################
    clf = joblib.load(f'{fi_dir}/pretrained_model/{model_name}_random_forest.joblib') 
    
    ####################################################
    # output prediction results
    ####################################################
    out = pdf.copy()
    pred_proba = clf.predict_proba(X)[:,-1]
    out['pred_proba'] = pred_proba
    fiName = 'TransferLearning_predictions.txt'
    if out_name == None:
        out.to_csv(f'{fi_dir}/prediction_results/{downstream_eval}/TransferLearning_predictions.txt', sep='\t', index=False)
    elif type(out_name) == str:
        fiName = out_name
        out.to_csv(f'{fi_dir}/prediction_results/{downstream_eval}/{out_name}.txt', sep='\t', index=False)
    else:
        raise TypeError("Provide correct outcome file name for 'out_name' parameter")
    print(f'Finished, {time.ctime()}')
    print(f"Prediction results available at : {fi_dir}/prediction_results/{downstream_eval}/{fiName}")
