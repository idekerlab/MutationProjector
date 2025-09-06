import pandas as pd
from collections import defaultdict
import os, time, sys
import numpy as np
import torch, math
import torch.nn as nn
import torch.nn.functional as F
import sklearn
from sklearn.preprocessing import *

# load mutational context (SBS96, SBS1536)
class load_mutational_context():
    def __init__(self, cohort):
        '''
        cohort: 'genie', 'TCGA or cbioportal source names
        '''
        self.cohort = cohort
        # file directory
        if self.cohort.upper() == 'GENIE':
            self.fi_dir = '/cellar/shared/j4kong/Project_Genie_cBioportal/mutational_signatures/analysis/output/SBS'
        elif self.cohort.upper() == 'TCGA':
            self.fi_dir = '/cellar/users/j4kong/Projects/datasets/TCGA_cBioportal/mutational_signatures'
        else:
            self.fi_dir = f'/cellar/users/j4kong/Projects/datasets/cbioportal/{self.cohort}/mutational_signatures/analysis/output'                
        
        
    def load_data(self, samples='default', context='SBS96', normalize=True):
        '''
        samples: 'default' or custom sample list
        context: 'SBS96', 'SBS1536'
        '''
        # load data
        df = pd.read_csv(f'{self.fi_dir}/Input_vcffiles.{context}.all', sep='\t')
        # contexts
        contexts = df['MutationType'].tolist()
        # normalize
        if normalize == True:
            for sample in df.columns[1:]:
                df[sample] = MinMaxScaler().fit_transform(np.array(df[sample].tolist()).astype(float).reshape(-1,1)).ravel()
        
        # out
        out = defaultdict(list)
        if type(samples) == str:
            if samples == 'default':
                out['sample'] = df.columns[1:]
                for c_idx, c in enumerate(contexts):
                    out[c] = df.values[c_idx][1:].astype(float)
                
        elif type(samples) == list or type(samples) == np.ndarray:
            out['sample'] = samples
            for c_idx, c in enumerate(contexts):
                tmp = []
                for sample in samples:
                    if not sample in df.columns:
                        tmp.append(0)
                    else:
                        tmp.append(df[sample].tolist()[c_idx])
                out[c] = tmp
            
        out = pd.DataFrame(data=out, columns=np.append(['sample'], contexts))
        return out

    
    

# using MESiCA classification criteria to identify dominant signatures
def load_dominant_signatures(cohort, signatures = ['APOBEC', 'Clock_SBS1', 'Clock_SBS5', 'MMR', 'POLE', 'Tobacco', 'UV']):
    '''
    cohort: 'TCGA' or cbioportal source names
    signatures: 'APOBEC', 'Clock_SBS1', 'Clock_SBS5', 'HRD', 'MMR', 'POLE', 'Tobacco', 'UV'
    '''
    cls_dic = {'APOBEC':['SBS2', 'SBS13'],
               'UV':['SBS7a', 'SBS7b', 'SBS7c', 'SBS7d', 'SBS38'],
               'Tobacco':['SBS4'],
               'HRD':['SBS3'],
               'Clock_SBS1':['SBS1'],
               'Clock_SBS5':['SBS5'],
               'POLE':['SBS10a', 'SBS10b'],
               'MMR':['SBS6', 'SBS14', 'SBS15', 'SBS20', 'SBS21', 'SBS26', 'SBS44']}
    SBS_signatures = []
    for sig in cls_dic.keys():
        SBS_signatures = list(set(SBS_signatures).union(cls_dic[sig]))

    # file location
    if cohort.upper() == 'TCGA':
        fi_dir = '/cellar/users/j4kong/Projects/datasets/TCGA_cBioportal/mutational_signatures'
    elif cohort != 'mixed_allen_2018':
        fi_dir = f'/cellar/users/j4kong/Projects/datasets/cbioportal/{cohort}/mutational_signatures/analysis/Assignment_Solution/Activities'
    elif cohort == 'mixed_allen_2018':
        fi_dir = '/cellar/users/j4kong/Projects/datasets/cbioportal/mixed_allen_2018/processed_files'
    
    # load data
    assert 'Assignment_Solution_Activities.txt' in os.listdir(fi_dir), f'No mutational signature file found from "{fi_dir}"'
    df = pd.read_csv(f'{fi_dir}/Assignment_Solution_Activities.txt', sep='\t')
    SBS_signatures = sorted(list(set(SBS_signatures) & set(df.columns)))
    
    # reformat df
    df = pd.DataFrame(data=df, columns=np.append(['Samples'], SBS_signatures))
    
    
    # compute relative contribution
    new_df = defaultdict(list)
    new_df['Samples'] = df['Samples'].tolist()
    df = df.set_index('Samples').astype(float)
    df = df.div(df.sum(axis=1), axis=0)
    
    # add relative contributions by signature type
    for sig in signatures:
        new_df[sig] = pd.DataFrame(data=df, columns=cls_dic[sig]).sum(axis=1).values
    new_df = pd.DataFrame(new_df, columns=np.append(['Samples'], signatures))

    
    # classify
    out = defaultdict(list); out['sample'] = new_df['Samples'].tolist()
    # non clock-like signatures
    for sig in signatures:
        if sig in ['Clock_SBS1', 'Clock_SBS5']: continue
        tmp = [0] * new_df.shape[0]
        
        for i, val in enumerate(new_df[sig].tolist()):
            # APOBEC, Tobacco, UV, MMR
            if sig in ['APOBEC', 'Tobacco', 'UV', 'MMR']:
                if val >= 0.3:
                    tmp[i] = 1
            # HRD
            elif sig == 'HRD':
                if val >= 0.5:
                    tmp[i] = 1
            # POLE
            elif sig == 'POLE':
                if val >= 0.2:
                    tmp[i] = 1
        # add to out
        out[sig] = tmp


    # clock-like signatures
    for i, sample in enumerate(out['sample']):
        non_clock = [out[sig][i] for sig in signatures if not 'Clock' in sig] 
        SBS1, SBS5 = new_df['Clock_SBS1'].tolist()[i], new_df['Clock_SBS5'].tolist()[i]
        # classify
        SBS1_cls, SBS5_cls = 0, 0
        if sum(non_clock) == 0:
            if (SBS5 < 0.4) and (SBS1 >= 0.4):
                SBS1_cls = 1
            elif (SBS1 < 0.4) and (SBS5 >= 0.4):
                SBS5_cls = 1
        # add to out
        out['Clock_SBS1'].append(SBS1_cls)
        out['Clock_SBS5'].append(SBS5_cls)
    
    # out --> dataframe
    out = pd.DataFrame(data=out, columns=np.append(['sample'], signatures))
    return out, new_df
                
            
def reorder_dominant_signatures(df, samples):
    out = defaultdict(list)
    df_columns = df.columns
    signatures = df.columns[1:]
    
    for sample in samples:
        out['sample'].append(sample)
        temp_values = [0] * len(signatures)
        if sample in df['sample'].tolist():
            temp_values = list(df.loc[df['sample']==sample,:].values[0][1:].astype(int))
        
        for sig_idx, sig in enumerate(signatures):
            out[sig].append(temp_values[sig_idx])
    return pd.DataFrame(data=out, columns=np.append(['sample'], signatures))
        