import numpy as np
import scipy.stats as stat
import torch, math
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import random
from itertools import *
import pandas as pd
from device_utils import resolve_device
#from tqdm import tqdm


## linear layer
class transform_layer(nn.Module):
    def __init__(self, num_signatures, num_features, cuda_device):
        super(transform_layer, self).__init__()
        self.num_signatures = num_signatures
        self.num_features = num_features
        self.cuda_device = resolve_device(cuda_device)
        
        # linear transformation
        self.DF_linear = nn.Sequential(
            nn.Linear(self.num_signatures, self.num_features).to(self.cuda_device),
            nn.LayerNorm(self.num_features).to(self.cuda_device),
            nn.ReLU().to(self.cuda_device))
            
        
    def forward(self, data_values):
        return self.DF_linear(data_values.to(self.cuda_device))

    
## tokenize special tokens
class tokenize_special_tokens(nn.Module):
    def __init__(self, num_features, num_bins, cuda_device):
        super(tokenize_special_tokens, self).__init__()
        self.num_features = num_features
        self.num_bins = num_bins
        self.cuda_device = resolve_device(cuda_device)
        
        # generate embedding layer
        self.token_emb = nn.Embedding(self.num_bins, self.num_features).to(self.cuda_device)
        
    def forward(self, values, apply_padding=False):
        # normalize input to a CPU float64 tensor (matches pandas' internal float64 bin-edge
        # math). Deliberately stays on CPU for this whole computation, regardless of
        # self.cuda_device: MPS (Apple Silicon) doesn't support float64 at all, and this is
        # cheap either way (num_bins+1 edges, not the full batch) -- only the final integer
        # bin indices get moved to the target device, for the embedding lookup.
        if type(values) == list or type(values) == np.ndarray:
            values_t = torch.as_tensor(values, dtype=torch.float64)
        else:
            values_t = values.detach().cpu().to(torch.float64).contiguous()

        # return embeddings
        if apply_padding == False:
            # bin values: replicates pandas.cut(bins=int)'s adaptive equal-width binning
            # (min/max of this call's values, 0.1%-of-range edge widening, right-closed
            # intervals) exactly, but without round-tripping the whole array through
            # GPU->CPU->pandas->GPU -- only the num_bins+1 edge scalars touch numpy.
            mn = values_t.min().item()
            mx = values_t.max().item()
            if mn == mx:
                adj_mn = mn - (0.001 * abs(mn) if mn != 0 else 0.001)
                adj_mx = mx + (0.001 * abs(mx) if mx != 0 else 0.001)
                edges_np = np.linspace(adj_mn, adj_mx, self.num_bins + 1)
            else:
                edges_np = np.linspace(mn, mx, self.num_bins + 1)
                edges_np[0] -= (mx - mn) * 0.001
            boundaries = torch.from_numpy(edges_np[1:-1])
            binned = torch.bucketize(values_t, boundaries, right=False)
            # return embeddings
            out_emb = self.token_emb(binned.to(self.cuda_device))
        elif apply_padding == True:
            # NOTE (pre-existing bug, out of scope for this refactor, flagged not fixed):
            # padding_emb is instantiated fresh on every call instead of being a persistent
            # __init__ submodule, so it is randomly re-initialized and never trained.
            padding_emb = nn.Embedding(1, self.num_features, padding_idx=0).to(self.cuda_device)
            binned = torch.zeros(values_t.shape[0], dtype=torch.long).to(self.cuda_device)
            out_emb = padding_emb(binned)
        return out_emb
            


##=====================================================================================================
## Generate minibatch
def return_indices(num_sample, batch_size, random_state, drop_last):
    output = []
    random.seed(random_state)
    ys = np.arange(num_sample)
    random.shuffle(ys)
    size = len(ys)//batch_size
    leftovers = ys[size*batch_size:]
    for i in range(size):
        output.append(ys[i*batch_size:(i+1)*batch_size])
    if len(leftovers) > 0:
        if drop_last == False:
            output.append(leftovers)
    return output



def batch_index(num_sample, batch_size, random_state=42, drop_last=True, sample_types=False):
    '''
    Example: random_state = 42; num_sample = 64; batch_size = 32
    '''
    if sample_types == False:
        output = return_indices(num_sample, batch_size, random_state, drop_last)
    
    elif type(sample_types) == list:
        # raise error
        if not num_sample == len(sample_types):
            print('num_sample not equal to sample_types')
            raise ValueError
        # batch index
        output = []
        tissue_types = sorted(list(set(sample_types)))
        for tissue_type in tissue_types:
            # tissue indices
            tissue_idx = []
            for idx, sample_type in enumerate(sample_types):
                if sample_type == tissue_type:
                    tissue_idx.append(idx)
            # minibatch
            while len(tissue_idx) >= batch_size:
                sampled = random.sample(tissue_idx, batch_size)
                output.append(sampled)
                tissue_idx = [idx for idx in tissue_idx if idx not in sampled]
            if drop_last == False:
                output.append(tissue_idx)
    return output



##=====================================================================================================
## Tokenize input and generate gene embeddings
def convert_mutations(X):
    # vocab index == binary value of each gene's feature tuple, MSB first
    # (itertools.product([0,1], repeat=k) enumerates in that exact order)
    k = X.shape[-1]
    weights = 2 ** torch.arange(k - 1, -1, -1)
    return (X.long() * weights).sum(dim=-1)



class tokenizer(nn.Module):
    def __init__(self, num_genes, num_features, cuda_device, num_gene_features=3, input_genes=[], cls_token=False, mask_nonzeros=False, gene_feature_index_to_mask=0):
        '''
        ##==============
        Input parameters
        N_genes, N_features
        '''
        super(tokenizer, self).__init__()
        self.num_genes = num_genes
        self.num_features = num_features
        self.num_gene_features=num_gene_features
        self.cuda_device = resolve_device(cuda_device)
        self.cls_token = cls_token
        self.mask_nonzeros = mask_nonzeros
        self.num_gene_features = num_gene_features
        self.gene_feature_index_to_mask = gene_feature_index_to_mask
        
        
        # all mutation combinations
        self.alt_combo = list(product([0, 1], repeat=self.num_gene_features))

        # vocab
        self.vocab = {}
        for value, key in enumerate(self.alt_combo):
            self.vocab[key] = value
        self.vocab['masked'] = value + 1

        
        ## embedding layers
        # gene embedding
        self.gene_embedding = nn.Embedding(self.num_genes, self.num_features).to(self.cuda_device)
        # use cls token
        if type(self.cls_token) == torch.nn.modules.sparse.Embedding:
            self.cls_token = self.cls_token.to(self.cuda_device)
            self.gene_embedding = nn.Embedding.from_pretrained( torch.cat((self.gene_embedding.weight, self.cls_token.weight), dim=0), freeze=False)
            
        # mutation embedding
        self.mut_embedding = nn.Embedding(len(self.vocab.keys()), self.num_features).to(self.cuda_device)
    
   
    
    def return_gene_embedding(self, num_samples):
        N_gene_emb = self.num_genes
        if type(self.cls_token) == torch.nn.modules.sparse.Embedding:
            N_gene_emb += 1
        out = self.gene_embedding(torch.tensor(np.arange(N_gene_emb)).to(self.cuda_device)).unsqueeze(0).repeat(num_samples, 1, 1)
        return out

    
    def return_mut_embedding(self, X_converted, mask_percentage, test_geneset):
        # create null mut embedding
        N_gene_emb = self.num_genes
        if type(self.cls_token) == torch.nn.modules.sparse.Embedding:
            N_gene_emb += 1        

        out = torch.zeros(N_gene_emb, self.num_features, requires_grad=True).to(self.cuda_device)
        out = out.unsqueeze(0).repeat(X_converted.shape[0], 1, 1)
        positions_to_mask = []
        
        # masking positions
        if test_geneset == False:
            num_to_mask = self.num_genes * mask_percentage // 100
            if self.mask_nonzeros == False:
                positions_to_mask = random.sample(range(self.num_genes), num_to_mask)
            elif self.mask_nonzeros == True:
                candidate_positions = []
                # nonzero tokens
                nonzero_tokens = []
                for key in vocab.keys():
                    if key != 'masked':
                        if key[self.gene_feature_index_to_mask] == 1:
                            nonzero_tokens.append(vocab[key])
                # find genes with at least one mutations across minibatch
                mask = torch.isin(X_converted, torch.tensor(nonzero_tokens))
                matching_columns = torch.any(mask, dim=0)
                candidate_positions = list(torch.nonzero(matching_columns, as_tuple=False).squeeze().numpy())
                positions_to_mask = random.sample(candidate_positions, np.min([num_to_mask, len(candidate_positions)]))
                

        # testing custom genes
        else:
            assert type(test_geneset)==list, 'provide a list of gene indices'
            positions_to_mask = test_geneset
            num_to_mask = len(positions_to_mask)
            
        # create mutation embeddings (one batched lookup instead of a per-(value,gene) loop)
        num_gene_cols = X_converted.shape[1]
        mut_emb_all = self.mut_embedding(X_converted.to(self.cuda_device))
        out[:, :num_gene_cols, :] = mut_emb_all

        # masking (applied after, so masked positions always end up with masked_emb)
        if num_to_mask > 0:
            masked_emb = self.mut_embedding(torch.tensor(self.vocab['masked']).to(self.cuda_device))
            out[:,positions_to_mask,:] = masked_emb.repeat(X_converted.shape[0], 1, 1)

        return out, positions_to_mask
        
        
    
    def forward(self, X_converted, mask_percentage, test_geneset):
        gene_emb = self.return_gene_embedding(X_converted.shape[0])
        mut_emb, positions_to_mask = self.return_mut_embedding(X_converted, mask_percentage, test_geneset)
        out = gene_emb + mut_emb
        return out, positions_to_mask




##=====================================================================================================
## merge data
def merge_data(mdf, cna, cnd, use_cancer_types=True):
    # merge data
    #common samples
    common_samples = sorted(list( set(mdf['sample'].tolist()) & set(cna['sample'].tolist()) & set(cnd['sample'].tolist()) ))

    #reorder data
    mdf = mdf.loc[mdf['sample'].isin(common_samples),:].sort_values(by='sample')
    cna = cna.loc[cna['sample'].isin(common_samples),:].sort_values(by='sample')
    cnd = cnd.loc[cnd['sample'].isin(common_samples),:].sort_values(by='sample')


    #ctypes
    col = ['sample']
    ctypes = []
    if use_cancer_types:
        ctypes = mdf['cancer_type'].tolist()
        col.append('cancer_type')


    #numpy
    mdf = mdf.set_index(col).values.T
    cna = cna.set_index(col).values.T
    cnd = cnd.set_index(col).values.T

    #convert data types
    X1 = torch.from_numpy(mdf).type(torch.float).T
    X2 = torch.from_numpy(cna).type(torch.float).T
    X3 = torch.from_numpy(cnd).type(torch.float).T
    
    # stack
    X = torch.stack((X1, X2, X3), dim=2)
    return X, np.array(common_samples), np.array(ctypes)


def merge_common_samples(df1, df2):
    common_samples = sorted(list(set(df1['sample'].tolist()) & set(df2['sample'].tolist())))
    df1 = df1.loc[df1['sample'].isin(common_samples),:].sort_values(by='sample')
    df2 = df2.loc[df2['sample'].isin(common_samples),:].sort_values(by='sample')
    return df1, df2
    