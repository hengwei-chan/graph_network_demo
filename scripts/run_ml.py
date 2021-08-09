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

from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

import graph_networks
import graph_networks.config
from graph_networks.utilities import ColumnsAction,readPickleGraphs,make_batch,atomgraphToNonGraphRepresentation,fit_model,test_model,getDataFromText,write_out


# =============================================================================
# GLOBAL FIELDS
# =============================================================================

PROJECT_PATH = Path(os.path.dirname(graph_networks.__file__)).parent.absolute()

# =============================================================================
# Methods
# =============================================================================

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

def testModel(feature_test,true_values,log_model,std,args):

    test_n_times = args.predict_n_times
    output_path = args.output_path

    log_loss=list()
    log_r2=list()
    predictions = list()
    for i in range(0,test_n_times): 
        predictions.append(test_model(feature_test,log_model,std))
        log_loss.append(np.math.sqrt(mean_squared_error(true_values,predictions[0])))
        log_r2.append(r2_score(true_values,predictions[0]))
    
    ci = 0.95
    log_loss_single = np.mean(log_loss)
    log_r2_single = np.mean(log_r2)
    if test_n_times > 1:
        lower_lim_log,upper_lim_log = np.quantile(log_loss, [0.025,0.025+ci], axis=0)
        #r2
        lower_lim_log_r2,upper_lim_log_r2 = np.quantile(log_r2, [0.025,0.025+ci], axis=0)

    write_out(output_path,'ml_predictions',
        str(args.log_types)+': '+str(np.round(log_loss_single,4))+' low_'+str(args.log_types)+': '+str(np.round(lower_lim_log,4))+' up_'+str(args.log_types)+': '+str(np.round(upper_lim_log,4))+
        ' log_r2_'+str(args.log_types)+': '+str(np.round(log_r2_single,4))+' low_r2_'+str(args.log_types)+': '+str(np.round(lower_lim_log_r2,4))+' up_r2_'+str(args.log_types)+': '+str(np.round(upper_lim_log_r2,4))+
        ' n_predict_boot: '+str(test_n_times)+' algo_typ: '+str(args.log_types)+' fp_types: '+str(args.fp_types)+
        ' n_bits: '+str(args.n_bits)+' radius: '+str(args.radius)+' include_descriptors: '+str(args.include_descriptors),
        0)
    return 

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

    log_model,std = fit_model(feature_train,logs_train,algo)

    testModel(feature_test,logs_test,log_model,std,args)

    return

# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser("ML Prediction Tool",description="Uses graph instances to predict different properties.")
    
    parser.add_argument('--output_path',required=True,type=str,
    help='Output path where the predictions are saved. Name of the file is "ml_predictions.txt" .')

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