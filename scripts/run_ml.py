# =============================================================================
# MODULE DOCSTRING
# =============================================================================

"""
This scripts is used to train and test the ML models using pickled graphs.
"""

# =============================================================================
# GLOBAL IMPORTS
# =============================================================================
import logging
import os
import sys
import numpy as np
from pathlib import Path
import random
import argparse
import shutil
from importlib import reload  
import pickle5 as pickle


import graph_networks
import graph_networks.config
from graph_networks.config import BasicModelConfig,Model1Config,DGINConfig,FrACConfig,MLConfig,Config
from graph_networks.utilities import ColumnsAction,readPickleGraphs,make_batch,atomgraphToNonGraphRepresentation,fit_model,test_model,getDataFromText


# =============================================================================
# GLOBAL FIELDS
# =============================================================================

PROJECT_PATH = Path(os.path.dirname(graph_networks.__file__)).parent.absolute()

# =============================================================================
# Methods
# =============================================================================


def get_data(config,seed,split):
    '''
    get the pickled data and make batches.\n
    Input:\n
        config (dataclass): the config defined in the input.\n
    Return:\n
        (list, list,list): train, evaluate and test batches of the graph instances. 

    '''
    # read test set and make batches
    test_batches = readPickleGraphs(config.test_data_dir,batch_size=config.batch_size)
    # read train/eval set and make batches
    train_eval_batches = readPickleGraphs(config.train_data_dir,batch_size=config.batch_size)
    # get train/eval split batches
    random.seed(seed)
    random.shuffle(train_eval_batches)
    train_batches = train_eval_batches[:int(len(train_eval_batches)*split)]
    eval_batches = train_eval_batches[int(len(train_eval_batches)*split):]

    return (train_batches,eval_batches), test_batches

def get_data_combined(config,validation_split,seed):

    # read the two test sets
    data_test = readPickleGraphs(config.test_data_dir,make_batches=False)
    add_data_test = readPickleGraphs(config.add_test_data_dir,make_batches=False)
    # combine test sets
    data_test.extend(add_data_test)
    # batch the whole test set
    test_batches = make_batch(data_test,config.batch_size)
    # read the two train/eval sets
    data_train = readPickleGraphs(config.train_data_dir,make_batches=False)
    add_data_train = readPickleGraphs(config.add_train_data_dir,make_batches=False)
    #### get train/eval data
    random.seed(seed)
    train_one = data_train[:int(len(data_train)*validation_split)]
    eval_one = data_train[int(len(data_train)*validation_split):]
    random.seed(seed)
    random.shuffle(add_data_train)
    train_two = add_data_train[:int(len(add_data_train)*validation_split)]
    eval_two = add_data_train[int(len(add_data_train)*validation_split):]
    # combine train/eval sets
    eval_one.extend(eval_two)
    train_one.extend(train_two)
    # batch the train/eval sets
    eval_batches = make_batch(eval_one,config.batch_size)
    train_batches = make_batch(train_one,config.batch_size)


    return (train_batches,eval_batches), test_batches

def prepare_ml_data(atom_graphs,config):
    '''
    prepares the AtomGraph instances for a ML application.
    Uses the defined ML Algorithm with its specifications for the molecular representation
    defined in the config file.\n
    Input:\n
        config (dataclass): the main config instance. \n
        atom_graphs (list of AtomGraph batches): AtomGraph instances to be prepared \n
    Returns:\n
        list of (properties_logD, properties_logS, properties_logP): instances for ML algorithms \n
        list of (properties_logD,properties_logS,properties_logP): correpsonding properties for ML algorithms \n
    '''

    representations_logD,representations_logP,representations_logS,representations_other = list(),list(),list(),list()
    properties_logD, properties_logP, properties_logS,properties_other = list(),list(),list(),list()
    # delete after:
    names_logD, names_logS, names_logP,names_other = list(),list(),list(),list()
    #
    for batch in atom_graphs:
        for graph in batch:
            if config.model1_config.include_logD:
                if graph.properties['logd']:
                    representations_logD.extend(atomgraphToNonGraphRepresentation(graph,config))
                    properties_logD.append(graph.properties['logd'])
                    # delete after:
                    names_logD.append(graph.name)
            if config.model1_config.include_logP:
                if graph.properties['logp']:
                    representations_logP.extend(atomgraphToNonGraphRepresentation(graph,config))
                    properties_logP.append(graph.properties['logp'])
                    # delete after:
                    names_logP.append(graph.name)
            if config.model1_config.include_logS:
                if graph.properties['logs']:
                    representations_logS.extend(atomgraphToNonGraphRepresentation(graph,config))
                    properties_logS.append(graph.properties['logs'])
                    # delete after:
                    names_logS.append(graph.name)
            if config.model1_config.include_other:
                if graph.properties['other']:
                    representations_other.extend(atomgraphToNonGraphRepresentation(graph,config))
                    properties_other.append(graph.properties['other'])

    return (representations_logD, representations_logS, representations_logP,representations_other),(properties_logD,properties_logS,properties_logP,properties_other),(names_logD, names_logS, names_logP,names_other)

def fitMLModels(config,data,properties):
    '''

    '''
    logd_model,logs_model,logp_model,other_model = None, None, None, None
    std_d,std_s,std_p,std_other = None, None, None, None
    if config.model1_config.include_logD:
        logd_model,std_d = fit_model(data[0],properties[0],config.ml_config.algorithm)
    if config.model1_config.include_logS:
        logs_model,std_s = fit_model(data[1],properties[1],config.ml_config.algorithm)
    if config.model1_config.include_logP:
        logp_model,std_p = fit_model(data[2],properties[2],config.ml_config.algorithm)
    if config.model1_config.include_other:
        other_model,std_other = fit_model(data[3],properties[3],config.ml_config.algorithm)

    return (logd_model,logs_model,logp_model,other_model),(std_d,std_s,std_p,std_other)

# =============================================================================
# Main Run Method
# =============================================================================

def run(args):
    '''
    The main method.
    runs the ML training and testing.
    '''
    log_type = args.log_types
    algo = args.algo_types
    fp_type = args.fp_types
    n_bits = args.n_bits
    radius = args.radius
    predict_n_times = args.predict_n_times

    if n_bits > 2049 or int(n_bits) < 0:
        print("Please define a valid nr of bits - maximum 2048, minimum 0. ")
        return 
    if int(radius) > 4:
        print("Please define a valid radius - maximum 4, minimum 1. ")
        return
    feature_train,feature_eval,feature_test = list(),list(),list()
    #train
    smiles_train, logs_train = getDataFromText("/home/owieder/projects/graph_networks/data/processed/ml/train_"+log_type+".csv")
    for smiles in smiles_train:
        feature_train.extend(atomgraphToNonGraphRepresentation(smiles,args))
    #eval
    smiles_eval, logs_eval = getDataFromText("/home/owieder/projects/graph_networks/data/processed/ml/eval_"+log_type+".csv")
    for smiles in smiles_eval:
        feature_eval.extend(atomgraphToNonGraphRepresentation(smiles,args))
    #test
    smiles_test, logs_test = getDataFromText("/home/owieder/projects/graph_networks/data/processed/ml/test_"+log_type+".csv")
    for smiles in smiles_test:
        feature_test.extend(atomgraphToNonGraphRepresentation(smiles,args))
    
    logd_model,std_d = fit_model(feature_train,logs_train,algo)

    return

# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser("ML Prediction Tool",description="Uses graph instances to predict different properties.")
    
    parser.add_argument('--log_types',required=True,
    help='Possible log types are: "logd", "logp" or "logs" - these types are then retrived from the txt files.',
    choices=['logs', 'logp', 'logd'])
    
    parser.add_argument('--algo_types',required=True,
    help='Possible algorithm types are: "svm", "rf" or "knn" - these types are then used to train and test '+
        'the data.',choices=['KNN', 'SVM', 'RF'])
    
    parser.add_argument('--fp_types',required=True,
    help='Possible fingerprint types are: "maccs" or "ecfp" - these types are then used as representation for '+
        'the data.',choices=['MACCS', 'ECFP'])

    parser.add_argument('--n_bits',required=False,type=int,
    help='Number of bits for the ECFP fingerprint. Maximum 2048. Default 2048.',default=2048)

    parser.add_argument('--radius',required=False,type=int,
    help='Radius defining the ECFP fingerprint. Default 4.',default=4)

    parser.add_argument('--include_descriptors',required=False,type=int,
    help='Flag to include all RDKit descriptors in addition to the fingerprint. "1" means include. '+
    '"0" means exclude. Default 1 - include!',default=1,choices=[1, 0])

    parser.add_argument('--predict_n_times',required=False,type=int,
    help='Bootstrapping iterations. How often should the test test be predicted - leave one out and replacement.' 
    ,default=100)

    
    args = parser.parse_args()

    run(args)
    
    print("Finished")