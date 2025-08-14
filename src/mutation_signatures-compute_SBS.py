#!/usr/bin/env python
# coding: utf-8
# compute SBS
# In[3]:


from SigProfilerMatrixGenerator.scripts import SigProfilerMatrixGeneratorFunc as matGen
from SigProfilerAssignment import Analyzer as Analyze
import os, time, sys
from collections import defaultdict
import pandas as pd
from tqdm import tqdm


# In[2]:


fldr_list = sorted(os.listdir('/cellar/users/j4kong/Projects/datasets/cbioportal'))


# In[9]:


fldr_list


# In[20]:


# exclude_list = ['BER_deficiency_signatures',
#                 'Chemotherapy_signatures',
#                 'Immunosuppressants_signatures'
#                 'Treatment_signatures'
#                 'AA_signatures',
#                 'Colibactin_signatures',
#                 'Artifact_signatures',
#                 'Lymphoid_signatures']
exclude_list = ['Artifact_signatures']


# In[22]:


# create binarized mutation files 
for i, fldr in enumerate(fldr_list):
    print(f'{i+1}/{len(fldr_list)}, {fldr}, {time.ctime()}')

    # load mutation information if available
    df = []
    try:
        df = pd.read_csv(f'/cellar/users/j4kong/Projects/datasets/cbioportal/{fldr}/data_mutations.txt', sep='\t', low_memory=False, comment='#')
    except: continue
    # reformat data
    df = pd.DataFrame(data=df, columns=['Hugo_Symbol', 'Entrez_Gene_Id', 'Center', 'NCBI_Build', 'Chromosome',
         'Start_Position', 'End_Position', 'Strand', 'Variant_Classification',
         'Variant_Type', 'Reference_Allele', 'Tumor_Seq_Allele1',
         'Tumor_Seq_Allele2', 'dbSNP_RS', 'dbSNP_Val_Status', 'Tumor_Sample_Barcode'])
    df2 = df.loc[df['Variant_Type']=='SNP',:]; df2 = df2.reset_index(drop=True)
    
    # genome build
    genome_build = df2['NCBI_Build'].tolist()[0]
    if fldr == 'mixed_selpercatinib_2020':
        genome_build = 'GRCh37'
    
    # create folder
    os.makedirs(f'/cellar/users/j4kong/Projects/datasets/cbioportal/{fldr}/mutational_signatures', exist_ok=True)
    os.makedirs(f'/cellar/users/j4kong/Projects/datasets/cbioportal/{fldr}/mutational_signatures/analysis', exist_ok=True)
    # write maf to new folder
    df2.to_csv(f'/cellar/users/j4kong/Projects/datasets/cbioportal/{fldr}/mutational_signatures/analysis/data_mutations.maf', sep='\t', index=False)

    
    ### run SigProfiler 
    fi_path = f'/cellar/users/j4kong/Projects/datasets/cbioportal/{fldr}/mutational_signatures/analysis'
    # Matrix Generator
    # matrices = matGen.SigProfilerMatrixGeneratorFunc('mutational_signatures', 'GRCh37', fi_path, seqInfo=False)
    # Assignment
    print(genome_build)
    assign_out = Analyze.cosmic_fit(fi_path, fi_path, input_type='vcf', genome_build=genome_build,
                                   exclude_signature_subgroups=exclude_list) 


# In[ ]:





# In[ ]:




