#!/usr/bin/env python

# =============================================================================
# MODULE DOCSTRING
# =============================================================================

"""
This scripts is used to generate graphs from smiles for the D-GIN publication.
"""

# =============================================================================
# GLOBAL IMPORTS
# =============================================================================
import logging
log = logging.getLogger(__name__)

import pickle
import random
import os

from multiprocessing import Pool
from functools import partial
from pathlib import Path
import argparse

# from graphnets.AtomGraph import AtomGraph
import graph_networks
from graph_networks.AtomGraph import AtomGraph
from graph_networks.utilities import readChemblXls, CDPLmolFromSmiles
# from graphnets.utilities_gen import plot, pickleGraphs
# from graphnets.utilities_chem import _getAllowedSet


PROJECT_PATH = Path(os.path.dirname(graph_networks.__file__)).parent.absolute()

def multi(data_combined):
    try:
        col_names = data_combined[-1]
        data = data_combined[:-1]
        indices = [i for i, x in enumerate(data) if x == '']
        mol = CDPLmolFromSmiles(data[1],False)
        graph = AtomGraph()
        graph(mol)
        graph.setName(data[0])
        for i,property_name in enumerate(col_names[2:]):
            graph.setProperty(property_name,(float(data[2+i])))
    except Exception as e:
            print("Problem with",data[0],data[2],e)
            return
    return graph


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser("Graph Generation Tool",description="Uses xls files with the names, smiles and different properties in each column to generate pickled graph representation for the D-GIN publication. The xls file needs to contain in the first row the name/description for eaach column. These names are taken for the property names.")
    parser.add_argument('--input_file_path',required=True,help='The path to the xls file.',type=str)
    parser.add_argument('--output_path',required=True,help='The path to the output folder.')
    parser.add_argument('--columns',required=True,nargs='+', type=int,help='Select the column for the name, smiles and other properties. The first to entries here need to be the name and smiles! Other Property names are extraced from the first row. e.g. if names are in column 0, smiles in column 7 and logD/logS endpoints in column 8 and 3 then use --columns 0 7 8 3')
    parser.add_argument('--skip_rows',type=int,help='How many rows should be skipped in addition to the first row of names/descriptions. So e.g. --skip_rows 2 skips one additional row. Default = 1',default=1)    
    parser.add_argument('--sheet_index',type=int,help="Sheet_index (int): Which sheet should be adressed. Default: 0 ",default=0)    
    parser.add_argument('--n_entries',type=int,help="Number of entries to be considered in the xls file. Default: 10000 ",default=10000)    
    parser.add_argument('--n_processes',type=int,help="Number of processes used on your machine. Default: 3 ",default=3)    

    args = parser.parse_args()

    data = readChemblXls(path_to_xls=args.input_file_path,col_entries=args.columns,sheet_index=args.sheet_index,skip_rows=args.skip_rows,n_entries=args.n_entries)
    graph_list = []
    pool = Pool(processes=int(args.n_processes))
    graph_list = pool.map(partial(multi),data)
    pool.close()
    pool.join()
    graph_list = list(filter(None, graph_list)) 

    split= int(len(graph_list)*0.90)
    random.seed(1)
    random.shuffle(graph_list)

    logD_graph_list_train_eval = graph_list[:split]
    logD_graph_list_test = graph_list[split:]

    pickleGraphs(PROJECT_PATH+"data/processed/lipo/pickled/train_dgin4_all_logs/",logD_graph_list_train_eval,5)
    pickleGraphs(PROJECT_PATH+"data/processed/lipo/pickled/test_dgin4_all_logs/",logD_graph_list_test,5)
