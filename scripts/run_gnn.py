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
from graph_networks.utilities import ColumnsAction,make_batch

from graph_networks.config import BasicModelConfig,Model1Config,DGINConfig,Config

from graph_networks.m

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
        from other_config import BasicModelConfig,Model1Config,DGINConfig,Config
    else:
        from graph_networks.config import BasicModelConfig,Model1Config,DGINConfig,Config
    config = Config(*(BasicModelConfig,Model1Config,DGINConfig))

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
        return
    logging.info("Create paths")
    try:
        Path(config.basic_model_config.config_log_dir).mkdir(parents=True, exist_ok=config.basic_model_config.override_if_exists)
        Path(config.basic_model_config.tensorboard_log_dir).mkdir(parents=True, exist_ok=config.basic_model_config.override_if_exists)
        Path(config.basic_model_config.model_weights_dir).mkdir(parents=True, exist_ok=config.basic_model_config.override_if_exists)
        Path(config.basic_model_config.plot_dir).mkdir(parents=True, exist_ok=config.basic_model_config.override_if_exists)
        Path(config.basic_model_config.stats_log_dir).mkdir(parents=True, exist_ok=config.basic_model_config.override_if_exists)
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

def get_data(config):
    data_folder_train = config.train_data_dir
    data_files_train = [fn for fn in os.listdir(config.train_data_dir)]

    data_folder_test = config.test_data_dir
    data_files_test = [fn for fn in os.listdir(config.test_data_dir)]

    data_train = list()
    data_test = list()

    for fn in data_files_train:
        fn = os.path.join(data_folder_train, fn)
        with open(fn,'rb') as f:
            data_train.extend(pickle.load(f))

    for fn in data_files_test:
        fn = os.path.join(data_folder_test, fn)
        with open(fn,'rb') as f:
            data_test.extend(pickle.load(f))

    train_batches = make_batch(data_train,config.batch_size)       
    test_batches = make_batch(data_test,config.batch_size)
    return train_batches, test_batches

def get_data_combined(config,validation_split,seed):
    data_folder_train = config.train_data_dir
    data_files_train = [fn for fn in os.listdir(config.train_data_dir)]

    data_folder_test = config.test_data_dir
    data_files_test = [fn for fn in os.listdir(config.test_data_dir)]


    data_train = list()
    data_test = list()

    for fn in data_files_train:
        fn = os.path.join(data_folder_train, fn)
        with open(fn,'rb') as f:
            data_train.extend(pickle.load(f))

    for fn in data_files_test:
        fn = os.path.join(data_folder_test, fn)
        with open(fn,'rb') as f:
            data_test.extend(pickle.load(f))
    
    
    add_data_folder_train = config.add_train_data_dir
    add_data_files_train = [fn for fn in os.listdir(config.add_train_data_dir)]

    add_data_folder_test = config.add_test_data_dir
    add_data_files_test = [fn for fn in os.listdir(config.add_test_data_dir)]

    add_data_train = list()
    add_data_test = list()

    for fn in add_data_files_train:
        fn = os.path.join(add_data_folder_train, fn)
        with open(fn,'rb') as f:
            add_data_train.extend(pickle.load(f))

    for fn in add_data_files_test:
        fn = os.path.join(add_data_folder_test, fn)
        with open(fn,'rb') as f:
            add_data_test.extend(pickle.load(f))
    
    # combine test sets
    data_test.extend(add_data_test)
    
    #### get train/eval data
    random.seed(seed)
    random.shuffle(data_train)
    train_one = data_train[:int(len(data_train)*validation_split)]
    eval_one = data_train[int(len(data_train)*validation_split):]

    random.seed(seed)
    random.shuffle(add_data_train)
    train_two = add_data_train[:int(len(add_data_train)*validation_split)]
    eval_two = add_data_train[int(len(add_data_train)*validation_split):]


    eval_one.extend(eval_two)
    train_one.extend(train_two)

    train_batches = make_batch(train_one,config.batch_size)
    eval_batches = make_batch(eval_one,config.batch_size)    
    test_batches = make_batch(data_test,config.batch_size)
    return (train_batches,eval_batches), test_batches

# =============================================================================
# Main Run Method
# =============================================================================

def run(config_path=None):
    '''
    The main method.
    runs the GNN training or testing.
    '''
    config = None
    try:
        config = get_config(config_path)
        create_paths(config,config_path)
    except Ex as e:
        logging.error("Error during config loading or path creation due to "+str(e))

    train_data, test_data = list(),list()
    if config.basic_model_config.combined_dataset:
        train_data, test_data = get_data_combined(config.basic_model_config,
                config.model1_config.validation_split,config.model1_config.train_data_seed)
    else:
        train_data, test_data = get_data(config.basic_model_config)

    gnn = Model(config)
    # if config.model in 'model2':
    #     gnn = Model2(config)
    # if config.model in 'model3':
    #     gnn = Model3(config)
    # if config.model in 'model4':
    #     gnn = Model4(config)
    # if config.model in 'model5':
    #     gnn = Model5(config)
    # if config.model == 'model10':
    #     gnn = Model10(config)
    # if config.model == 'model11':
    #     gnn = Model11(config)

    # gnn.compile(loss=config.model1_config.lipo_loss_mse,
    #             optimizer=config.model1_config.optimizer)
    
    # if config.basic_model_config.test_model:
    #     gnn.load_weights(config.basic_model_config.test_model_weights_dir)
    #     from joblib import dump
    #     # dump(gnn,'/home/owieder/LOGD_DGIN.joblib')
    #     if config.model == 'model10':
    #         gnn.test(test_data,config.basic_model_config.test_model_epoch,None,save_weights_flag=False,
    #                 include_dropout=False,predict_n_times=1,simply_testing=False)
    #     else:
    #         gnn.test(test_data,config.basic_model_config.test_model_epoch,None,
    #             include_dropout=True,predict_n_times=1)
    #     return
    # if config.basic_model_config.encode_hidden:
    #     gnn.load_weights(config.basic_model_config.test_model_weights_dir)
    #     gnn.encode(test_data)
    #     return

    # if config.basic_model_config.retrain_model:
    #     if config.basic_model_config.model_name == config.basic_model_config.retrain_model_name:
    #         logging.exception("New model name should be different to old model name."+
    #         config.basic_model_config.model_name+config.basic_model_config.retrain_model_name)
    #         return
    #     gnn.load_weights(config.basic_model_config.retrain_model_weights_dir)
    # gnn.run(train_data, test_data)

    return

# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser("GNN Prediction Tool",description="Uses graph instances to predict different properties.")
    
    parser.add_argument('--config_path',
    help='Path to the config file folder. In this path the config file MUST be named "other_config.py".'+
    ' If not set, the default ./graph_networks/config.py is taken.')
    
    args = parser.parse_args()

    if args.config_path is None:
        decision = input('Are you sure you want to use the default ./graph_networks/config.py file '+
        'without setting your own config file path? ')
        while ('yes' not in decision.lower()) or ('configs' not in decision.lower()):
            decision = input('Please define either valid config file folder path '+
            '(path MUST include the "./reports/user_defined_configs/" path! The folder MUST include a config file'+
            ' named as "other_config.py") or confirm with "yes"' +
            ' to use the default settings: ')
            if 'user_defined_configs' in decision.lower():
                args.config_path = decision
                break
            if 'yes' in decision.lower():
                break

    run(args.config_path)
    
    print("Finished")