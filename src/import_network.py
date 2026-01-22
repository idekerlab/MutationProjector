import networkx as nx
from collections import defaultdict
import pandas as pd
import ndex2
import numpy as np
from pathlib import Path
from load_geneList import *
import torch



## import network
class load_network():
    def __init__(self):
        super(load_network, self).__init__()
        try: 
            self.fi_dir = Path(__file__).resolve().parents[1]
        except NameError:
            self.fi_dir = Path().resolve().parent

            
    def return_edges(self, network):
        '''
        Provide edge information for given network name and custom gene list
        -----------------------
        # Input
        network : 'genetic_interaction', 'GRN', 'STRING', 'physical_ppi', 'E3', 'phosphorylation', 'DDRAM', 'PCNET'

        -----------------------
        # Returns
        list of lists containing [gene1 index, gene2 index].
        shape (Num edges, 2)
        '''
        
        ##=================
        # load networks        
        # pt file
        out = defaultdict(list)
        if network.upper() in ['STRING', 'DDRAM', 'PCNET']:
            fiName = network.upper()
        else:
            fiName = f'{network}_expanded'

        edges = torch.load(f'{self.fi_dir}/data/networks/{fiName}.pt')
        return edges

