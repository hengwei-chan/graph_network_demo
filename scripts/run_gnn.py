# =============================================================================
# MODULE DOCSTRING
# =============================================================================

"""
This scripts is used to train and test the GNN models using pickled graphs.
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
from graph_networks.utilities import ColumnsAction,readPickleGraphs,make_batch,atomgraphToNonGraphRepresentation,fit_model,test_model
from graph_networks.model import BaseModel as Model
from graph_networks.model10 import Model10


# =============================================================================
# GLOBAL FIELDS
# =============================================================================

PROJECT_PATH = Path(os.path.dirname(graph_networks.__file__)).parent.absolute()

# =============================================================================
# Methods
# =============================================================================

def get_config(config_path):
    '''
    retrives the configs for the GNNs. For backwards compatibility, when certain fields
    are not present, they are being set to False.

    '''
    if config_path:
        sys.path.insert(0, config_path)
        import other_config
        reload(other_config)
        from other_config import BasicModelConfig,Model1Config,DGINConfig,FrACConfig,MLConfig,Config
    else:
        from graph_networks.config import BasicModelConfig,Model1Config,DGINConfig,FrACConfig,MLConfig,Config
    config = Config(*(BasicModelConfig,Model1Config,DGINConfig,FrACConfig,MLConfig))

    logging.basicConfig(filename=config.basic_model_config.log_dir, level=config.basic_model_config.verbosity_level)

    try:
        if config.model1_config.include_logS:
            print(config.basic_model_config.model_name ,' has include_logS')
            logging.info(config.basic_model_config.model_name +' has include_logS')
        else:
            print(config.basic_model_config.model_name ,' has include_logS as FALSE')
            logging.info(config.basic_model_config.model_name +' has include_logS as FALSE')
    except Exception as e:
        print(config.basic_model_config.model_name ,' does NOT have include_logS - is being added as FALSE.')
        logging.info(config.basic_model_config.model_name +' does NOT have include_logS - is being added as FALSE.')
        config.model1_config.include_logS = False
    try: # yes
        if config.model1_config.include_logD:
            print(config.basic_model_config.model_name ,' has include_logD')
            logging.info(config.basic_model_config.model_name +' has include_logD')
        else:
            print(config.basic_model_config.model_name ,' has include_logD as FALSE')
            logging.info(config.basic_model_config.model_name +' has include_logD as FALSE')
    except Exception as e:
        print(config.basic_model_config.model_name ,' does NOT have include_logD - is being added as True.')
        logging.info(config.basic_model_config.model_name +' does NOT have include_logD - is being added as True.')
        config.model1_config.include_logD = True
    try: # yes
        if config.model1_config.include_logP:
            print(config.basic_model_config.model_name ,' has include_logP')
            logging.info(config.basic_model_config.model_name +' has include_logP')
        else:
            print(config.basic_model_config.model_name ,' has include_logP as FALSE')
            logging.info(config.basic_model_config.model_name +' has include_logP as FALSE')
    except Exception as e:
        print(config.basic_model_config.model_name ,' does NOT have include_logP - is being added as True.')
        logging.info(config.basic_model_config.model_name +' does NOT have include_logP - is being added as True.')
        config.model1_config.include_logP = True
    try:
        if config.model1_config.include_other:
            print(config.basic_model_config.model_name ,' has include_other')
            logging.info(config.basic_model_config.model_name +' has include_other')
        else:
            print(config.basic_model_config.model_name ,' has include_other as FALSE')
            logging.info(config.basic_model_config.model_name +' has include_other as FALSE')
    except Exception as e:
        print(config.basic_model_config.model_name ,' does NOT have include_other - is being added as FALSE.')
        logging.info(config.basic_model_config.model_name +' does NOT have include_other - is being added as FALSE.')
        config.model1_config.include_other = False
    try: # yes
        if config.model1_config.best_evaluation_threshold is float:
            print(config.basic_model_config.model_name ,' has best_evaluation_threshold:',config.model1_config.best_evaluation_threshold)
            logging.info(config.basic_model_config.model_name +' has best_evaluation_threshold:'+config.model1_config.best_evaluation_threshold)
    except Exception as e:
        print(config.basic_model_config.model_name ,' does NOT have best_evaluation_threshold - is being added as 2.35.')
        logging.info(config.basic_model_config.model_name +' does NOT have best_evaluation_threshold - is being added as 2.35.')
        config.model1_config.best_evaluation_threshold = 2.75
    try: # yes
        if config.basic_model_config.combined_dataset:
            print(config.basic_model_config.model_name ,' has combined_dataset')
            logging.info(config.basic_model_config.model_name +' has combined_dataset')
    except Exception as e:
        print(config.basic_model_config.model_name ,' does NOT have combined_dataset - is being added as FALSE.')
        logging.info(config.basic_model_config.model_name +' does NOT have combined_dataset - is being added as FALSE.')
        config.basic_model_config.combined_dataset = False
        print('Additionally, the path is added as None.')
        logging.infoprint('Additionally, the path is added as None.')
        config.basic_model_config.add_train_data_dir = None
        config.basic_model_config.add_test_data_dir = None
    try: # yes
        if config.basic_model_config.model_type:
            print(config.basic_model_config.model_name ,' has model_type ',config.basic_model_config.model_type)
            logging.info(config.basic_model_config.model_name +' has model_type '+config.basic_model_config.model_type)
    except Exception as e:
        print(config.basic_model_config.model_name ,' does NOT have model_type - is being added as default model_type DGIN.')
        logging.info(config.basic_model_config.model_name +' does NOT have model_type - is being added as default model_type DGIN.')
        config.basic_model_config.model_type = 'DGIN'

    return config

def create_paths(config,config_path):
    '''
    Creates the paths for the GNN model if not already there under the given model_name
    (model weights, log file, tensorboard, plots, statistics, configs).
    '''
    # check if paths are there, otherwise create
    if config.basic_model_config.test_model:
        try:
            ########################################################################
            ########################################################################
            ######### Change this!
            Path(config.basic_model_config.plot_dir).mkdir(parents=True, exist_ok=False)
            # Path(config.basic_model_config.model_weights_dir+'eval/logd/').mkdir(parents=True, exist_ok=False)
            # Path(config.basic_model_config.model_weights_dir+'eval/logs/').mkdir(parents=True, exist_ok=False)
            # Path(config.basic_model_config.model_weights_dir+'eval/logp/').mkdir(parents=True, exist_ok=False)
            Path(config.basic_model_config.stats_log_dir).mkdir(parents=True, exist_ok=False)
            
            Path(config.basic_model_config.test_prediction_output_folder).mkdir(parents=True, exist_ok=False)
            # Path(config.basic_model_config.model_weights_dir+'test/logd/').mkdir(parents=True, exist_ok=False)
            # Path(config.basic_model_config.model_weights_dir+'test/logs/').mkdir(parents=True, exist_ok=False)
            # Path(config.basic_model_config.model_weights_dir+'test/logp/').mkdir(parents=True, exist_ok=False)
            # Path(config.basic_model_config.stats_log_dir+'test/').mkdir(parents=True, exist_ok=False)
            # Path(config.basic_model_config.stats_log_dir+'eval/').mkdir(parents=True, exist_ok=False)
        except FileExistsError as e:
            logging.error("The current model under"+config.basic_model_config.model_name+" exists already - either rename or set override_if_exists to 'True'")
            raise e
        logging.info("Paths created for"+config.basic_model_config.model_name)

        return
    logging.info("Create paths")
    try:
        Path(config.basic_model_config.config_log_dir).mkdir(parents=True, exist_ok=config.basic_model_config.override_if_exists)
        Path(config.basic_model_config.tensorboard_log_dir).mkdir(parents=True, exist_ok=config.basic_model_config.override_if_exists)
        Path(config.basic_model_config.plot_dir).mkdir(parents=True, exist_ok=config.basic_model_config.override_if_exists)
        Path(config.basic_model_config.model_weights_dir).mkdir(parents=True, exist_ok=config.basic_model_config.override_if_exists)
        # Path(config.basic_model_config.model_weights_dir+'eval/logs/').mkdir(parents=True, exist_ok=config.basic_model_config.override_if_exists)
        # Path(config.basic_model_config.model_weights_dir+'eval/logp/').mkdir(parents=True, exist_ok=config.basic_model_config.override_if_exists)
        Path(config.basic_model_config.stats_log_dir).mkdir(parents=True, exist_ok=config.basic_model_config.override_if_exists)
        
        Path(config.basic_model_config.test_prediction_output_folder).mkdir(parents=True, exist_ok=config.basic_model_config.override_if_exists)
        # Path(config.basic_model_config.model_weights_dir+'test/logd/').mkdir(parents=True, exist_ok=config.basic_model_config.override_if_exists)
        # Path(config.basic_model_config.model_weights_dir+'test/logs/').mkdir(parents=True, exist_ok=config.basic_model_config.override_if_exists)
        # Path(config.basic_model_config.model_weights_dir+'test/logp/').mkdir(parents=True, exist_ok=config.basic_model_config.override_if_exists)
        Path(config.basic_model_config.stats_log_dir+'test/').mkdir(parents=True, exist_ok=config.basic_model_config.override_if_exists)
        Path(config.basic_model_config.stats_log_dir+'eval/').mkdir(parents=True, exist_ok=config.basic_model_config.override_if_exists)
    except FileExistsError as e:
        logging.error("The current model under"+config.basic_model_config.model_name+" exists already - either rename or set override_if_exists to 'True'")
        raise e
    logging.info("Paths created for"+config.basic_model_config.model_name)

    if config_path:
        shutil.copyfile(config_path+'/other_config.py',config.basic_model_config.config_log_dir+config.basic_model_config.model_name)
    else:
        shutil.copyfile(config.basic_model_config.project_path+'/graph_networks/config.py',config.basic_model_config.config_log_dir+config.basic_model_config.model_name)
        shutil.copyfile(config.basic_model_config.project_path+'/graph_networks/config.py',config.basic_model_config.config_log_dir+'/other_config.py')

    logging.info("Fetched config data and created paths for "+config.basic_model_config.model_name)

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

def run(config_path=None):
    '''
    The main method.
    runs the GNN training or testing.
    '''
    config = None
    # try:
    config = get_config(config_path)
    create_paths(config,config_path)
    # except Exception as e:
    #     logging.error("Error during config loading or path creation due to "+str(e))

    train_data, test_data = list(),list()
    if config.basic_model_config.combined_dataset:
        train_data, test_data = get_data_combined(config.basic_model_config,
                        config.model1_config.validation_split,
                        config.model1_config.train_data_seed,
                        )
    else:
        train_data, test_data = get_data(config.basic_model_config,
                    config.model1_config.train_data_seed,
                    config.model1_config.validation_split
                    )

    gnn = Model(config)   
    # if config.model in 'model2':
    #     gnn = Model2(config)
    # if config.model in 'model3':
    #     gnn = Model3(config)
    # if config.model in 'model4':
    #     gnn = Model4(config)
    # if config.model in 'model5':
    #     gnn = Model5(config)
    if config.model == 'model10':
        gnn = Model10(config)
    if config.model == 'model11':
        gnn = Model10(config)

    gnn.compile(loss=config.model1_config.lipo_loss_mse,
                optimizer=config.model1_config.optimizer)

    #### for the docker:
    config.basic_model_config.consensus = False
    
    if config.basic_model_config.test_model:
        if config.basic_model_config.test_n_times < 1 or config.basic_model_config.save_predictions:
            config.basic_model_config.test_n_times = 1
            logging.debug("test_n_times is set to 1 as it was below 1 or save_predictions is True!")
        gnn.load_weights(config.basic_model_config.test_model_weights_dir)
        if config.basic_model_config.consensus:
            # changes names
            train_data_ml, train_properties_ml,train_names = prepare_ml_data(train_data[0],config)
            test_data_ml, test_properties_ml,test_names = prepare_ml_data(test_data,config)
            models,stds = fitMLModels(config,train_data_ml,train_properties_ml)
            # changes names
            gnn.test(test_data=test_data,
                    test_data_ml=(test_data_ml,test_properties_ml,test_names),
                    epoch=config.basic_model_config.test_model_epoch,
                    include_dropout=config.basic_model_config.include_dropout,
                    test_n_times=config.basic_model_config.test_n_times,
                    consensus=config.basic_model_config.consensus,
                    ml_models=models,stds=stds
                    )
        else:
            gnn.test(test_data=test_data,
                    epoch=config.basic_model_config.test_model_epoch,
                    include_dropout=config.basic_model_config.include_dropout,
                    test_n_times=config.basic_model_config.test_n_times
                    )
        return

    if config.basic_model_config.retrain_model:
        if config.basic_model_config.model_name == config.basic_model_config.retrain_model_name:
            logging.error("New model name should be different to old model name."+
            config.basic_model_config.model_name+config.basic_model_config.retrain_model_name)
            return
        gnn.load_weights(config.basic_model_config.retrain_model_weights_dir)
    gnn.run(train_data,test_data)

    return

# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser("GNN Prediction Tool",description="Uses graph instances to predict different properties.")
    
    parser.add_argument('--config_path',required=True,
    help='Path to the config file folder. In this path the config file MUST be named "other_config.py".'+
    'Define the mode there. If you want to train the model, create a new path under reports/configs/"model_name"/,'+
    ' copy paste a config file there with the name "other_config.py", define as wanted and run_gnn.py. For training set test_model=False. Otherwise to True.')
    
    args = parser.parse_args()

    # if args.config_path is None:
    #     decision = input('Are you sure you want to use the default ./graph_networks/config.py file '+
    #     'without setting your own config file path? ')
    #     while ('yes' not in decision.lower()) or ('configs' not in decision.lower()):
    #         decision = input('Please define either valid config file folder path '+
    #         '(path MUST include the "./reports/user_defined_configs/" path! The folder MUST include a config file'+
    #         ' named as "other_config.py") or confirm with "yes"' +
    #         ' to use the default settings: ')
    #         if 'user_defined_configs' in decision.lower():
    #             args.config_path = decision
    #             break
    #         if 'yes' in decision.lower():
    #             break

    run(args.config_path)
    
    print("Finished")