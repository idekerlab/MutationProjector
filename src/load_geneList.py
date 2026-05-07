import pandas as pd
import numpy as np
from collections import defaultdict
import os, time, sys
from pathlib import Path


def load_genes(gset='MSKIMPACT468', convert_IDs=True):
    # fi_dir
    try: 
        fi_dir = Path(__file__).resolve().parents[1]
    except NameError:
        fi_dir = Path().resolve().parent
    # load genelist
    if (gset == 'MSK-IMPACT468') or (gset == 'MSKIMPACT468'):
        if gset == 'MSKIMPACT468':
            gset = 'MSK-IMPACT468'
        df = pd.read_csv(f'{fi_dir}/data/gene/%s.txt'%gset, sep='\t')
        geneList = sorted(list(set(df['gene'].tolist())))

        
    # convert alias gene IDs to primary gene IDs
    if convert_IDs == True:
        geneList = convert_geneIDs(geneList)
        geneList = sorted(geneList)
    return geneList


# uniprot database
def load_mapped_geneNames():
    out = {} # {synonym : unique IDs}
    # fi_dir
    try: 
        fi_dir = Path(__file__).resolve().parents[1]
    except NameError:
        fi_dir = Path().resolve().parent
    # NCBI synonymous gene IDs
    out_df = pd.read_csv(f'{fi_dir}/data/gene/geneIDs.txt', sep='\t')
    out = dict(zip(out_df['synonym'].tolist(), out_df['primary'].tolist()))
    return out
                



# convert gene IDs
def convert_geneIDs(geneList):
    out = []
    geneID = load_mapped_geneNames()
    for gene in geneList:
        new_gene = gene
        if gene in list(geneID.keys()):
            new_gene = geneID[gene]
        out.append(new_gene)
    return out



# convert pandas gene columns
def convert_dataframe_geneIDs(df):
    gene_col = df.columns
    new_gene_col = convert_geneIDs(gene_col)
    new_col = dict(zip(gene_col, new_gene_col))
    df = df.rename(columns=new_col)
    df = pd.DataFrame(data=df, columns=new_gene_col)
    return df
    
