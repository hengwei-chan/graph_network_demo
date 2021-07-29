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

import random
import os
from itertools import repeat
from multiprocessing import Pool
from functools import partial
from pathlib import Path
import argparse
import datetime

import graph_networks
from graph_networks.AtomGraph import AtomGraph
from graph_networks.utilities import readChemblXls, CDPLmolFromSmiles, pickleGraphs, LOG_LEVELS

# =============================================================================
# GLOBAL FIELDS
# =============================================================================

PROJECT_PATH = Path(os.path.dirname(graph_networks.__file__)).parent.absolute()

# =============================================================================
# Methods
# =============================================================================

def multi_threading(data_combined,featurization):
    ''' 
    PRIVATE METHOD 
    method for the pool instance used in various scripots (e.g. during graph generation). \n
    Input \n
        data_combined (tuple): tuple of two lists: first is a list of the data (name,smiles,properties), \n
             second is a list of property names. \n
    Returns: \n
        (AtomGraph): graph instances of the molecule. 
    '''
    try:
        property_names = data_combined[-1]
        data = data_combined[:-1]
        indices = [i for i, x in enumerate(data) if x == '']
        mol = CDPLmolFromSmiles(data[1],False)
        if mol is None:
            logging.debug("Could not process "+str(data[0])+" "+str(data[2])+" because of its multi comp!")
            return None
        graph = AtomGraph()
        graph(mol,featurization=featurization)
        graph.setName(data[0])
        graph.setSmiles(data[1])
        for i,property_name in enumerate(property_names[2:]):
            if 'logs' in property_name.lower():
                if float(data[2+i]) >0.0 or float(data[2+i]) < (-10.0):
                    return None
                graph.setProperty(property_name.lower(),(float(data[2+i])+10.0))
            elif 'logp' in property_name.lower():
                graph.setProperty(property_name.lower(),(float(data[2+i])+3.0))
            elif 'logd' in property_name.lower():
                graph.setProperty(property_name.lower(),(float(data[2+i])+1.60))
            else:
                graph.setProperty('other',float(data[2+i]))
        if not 'logd' in graph.properties:
            graph.setProperty('logd',False)
        if not 'logp' in graph.properties:
            graph.setProperty('logp',False)
        if not 'logs' in graph.properties:
            graph.setProperty('logs',False)
        if not 'other' in graph.properties:
            graph.setProperty('other',False)
    except Exception as e:
            logging.debug("Could not process "+str(data[0])+" "+str(data[2])+" because of "+str(e))
            return None
    return graph

# =============================================================================
# Main Run Method
# =============================================================================

def run(args):
    '''
    The main method for the graph generation.
    runs the GNN training or testing.
    '''

    try:
        if not os.path.isdir(args.output_path_train):
            raise FileExistsError("The output path does not exist - please create one with the corresponding name.")

        logging.debug("Start read FILE and generate data!")
        data = readChemblXls(path_to_xls=args.input_file_path,col_entries=args.columns,sheet_index=args.sheet_index,skip_rows=args.skip_rows,n_entries=args.n_entries)
        logging.debug("Finished FILE and data reading with overall nr of entries: "+str(len(data)))
        print("Finished FILE and data generation with overall nr of entries: "+str(len(data)))
        graph_list = []
        print("Start graph generation.")
        pool = Pool(processes=int(args.n_processes))
        logging.debug("Start muli threading and graph list generation!")
        graph_list = pool.starmap(partial(multi_threading),zip(data, repeat(args.featurization)))
        logging.debug("Finished muli threading and graph list generation!")
        pool.close()
        pool.join()
        graph_list = list(filter(None, graph_list))
        print("Finished graph generation with overall nr of entries: "+str(len(graph_list)))
        logging.info("Finished graph generation with overall nr of entries: "+str(len(graph_list)))

        split= int(len(graph_list)*args.train_test_split)
        random.seed(1)
        random.shuffle(graph_list)

        
        # logd_train = list()
        # with open('/home/owieder/projects/old_logd_train.txt') as f:
        #     lines = f.readlines()
        #     for line in lines:
        #         logd_train.append(line.split(' ')[0])
        # logd_test = list()
        # with open('/home/owieder/projects/old_logd_test.txt') as f:
        #     lines = f.readlines()
        #     for line in lines:
        #         logd_test.append(line.split(' ')[0])
        # sorted_graph_list_train = list()
        # sorted_graph_list_test = list()
        # for name in logd_train:
        #     for graph in graph_list:
        #         if name == graph.name:
        #             sorted_graph_list_train.append(graph)
        # for name in logd_test:
        #     for graph in graph_list:
        #         if name == graph.name:
        #             sorted_graph_list_test.append(graph)        
        

        # logs_train = list()
        # with open('/home/owieder/projects/old_logs_train.txt') as f:
        #     lines = f.readlines()
        #     for line in lines:
        #         logs_train.append(line.split(' ')[0])
        # logs_test = list()
        # with open('/home/owieder/projects/old_logs_test.txt') as f:
        #     lines = f.readlines()
        #     for line in lines:
        #         logs_test.append(line.split(' ')[0])

        # sorted_graph_list_train = list()
        # sorted_graph_list_test = list()
        # for name in logs_train:
        #     for graph in graph_list:
        #         if name == graph.name:
        #             sorted_graph_list_train.append(graph)
        # for name in logs_test:
        #     for graph in graph_list:
        #         if name == graph.name:
        #             sorted_graph_list_test.append(graph)

        logD_graph_list_train_eval = graph_list[:split]
        logD_graph_list_test = graph_list[split:]
        logging.info("Train/Evaluation graph list length: "+str(len(logD_graph_list_train_eval)))
        logging.info("Test graph list length: "+str(len(logD_graph_list_test)))

        print("Start pickling...")
        logging.debug("Start pickling graph lists!")
        pickleGraphs(args.output_path_train,logD_graph_list_train_eval,args.pickle_split)
        logging.debug("Finished train/eval pickling!")
        pickleGraphs(args.output_path_test,logD_graph_list_test,args.pickle_split)
        logging.debug("Finished test pickling!")

    except Exception as e:
        logging.error("Could not finish the graph generation due to "+str(e))

# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Graph Generation Tool",description="Uses xls files with the names, smiles and different properties in each column to generate pickled graph representation for the D-GIN publication. The xls file needs to contain in the first row the name/description for eaach column. These names are taken for the property names.")
    parser.add_argument('--input_file_path',required=True,help='REQUIRED! The path to the xls file.',type=str)
    parser.add_argument('--output_path_train',required=True,help='REQUIRED! The path to the output folder FOR TRAINING.')
    parser.add_argument('--output_path_test',required=True,help='REQUIRED! The path to the output folder FOR TESTING.')
    parser.add_argument('--columns',required=True,nargs='+', type=int,help='REQUIRED! Select the column for the name, smiles and other properties. The first to entries here need to be the name and smiles! Other Property names are extraced from the first row. e.g. if names are in column 0, smiles in column 7 and logD/logS endpoints in column 8 and 3 then use --columns 0 7 8 3')
    parser.add_argument('--log_dir',help='REQUIRED! The log directory for the graph generation script.',required=True)
    parser.add_argument('--featurization',type=str,help="Define the featurization type of the graph. Allowed featurizations are: " +
     "'DMPNN','DGIN', 'DGIN3', 'DGIN4', 'DGIN5', 'DGIN6', 'DGIN7', 'DGIN8', 'DGIN9' ")    
    parser.add_argument('--skip_rows',type=int,help='How many rows should be skipped in addition to the first row of names/descriptions. So e.g. --skip_rows 2 skips one additional row. Default = 1',default=1)    
    parser.add_argument('--sheet_index',type=int,help="Sheet_index (int): Which sheet should be adressed. Default: 0 ",default=0)    
    parser.add_argument('--n_entries',type=int,help="Number of entries to be considered in the xls file. Default: 10000 ",default=10000)    
    parser.add_argument('--n_processes',type=int,help="Number of processes used on your machine. Default: 3 ",default=3)    
    parser.add_argument('--train_test_split',type=float,help="Split for training/testing. e.g. 0.9 means that 90 percent of the " +
     "data is taken as training, the rest (10 percent) as testing data. Default: 0.9 ",default=0.9)   
    parser.add_argument('--log_verbosity', default=2, type=int,
                    help="Verbosity (between 1-4 occurrences with more leading to more "
                         "verbose logging). CRITICAL=0, ERROR=1, WARN=2, INFO=3, "
                         "DEBUG=4 - Default=3")

    parser.add_argument('--pickle_split',type=int,help="Number of pickled data instances. Default: 5 ",default=5)    

    args = parser.parse_args()

    #Path(args.log_dir).mkdir(parents=True, exist_ok=False)

    logging.basicConfig(filename=args.log_dir+'/gen_graph.log', level=LOG_LEVELS[args.log_verbosity])

    logging.info("NEW!!!! Start graph generation. "+ datetime.datetime.now().strftime('%D:%H.%f')[:-4])
    logging.info("input_file_path:"+str(args.input_file_path))
    logging.info("output_path_train:"+str(args.output_path_train))
    logging.info("output_path_test:"+str(args.output_path_test))
    logging.info("train_test_split:"+str(args.train_test_split))
    logging.info("featurization:"+str(args.featurization))

    print("Start graph generation - might take some time, depending on the amount of data!")
    run(args)
    print("Finished! For more details look into the log file: "+str(args.log_dir))
    logging.info("Finished graph generation. "+ datetime.datetime.now().strftime('%D:%H.%f')[:-4]+'\n')