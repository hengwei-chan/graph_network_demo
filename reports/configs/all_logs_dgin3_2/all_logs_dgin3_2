#!/usr/bin/env python

# =============================================================================
# MODULE DOCSTRING
# =============================================================================

"""

"""

# =============================================================================
# GLOBAL IMPORTS
# =============================================================================

from dataclasses import dataclass, field
from typing import List

import tensorflow as tf
from graph_networks.utilities import *
import logging
import os




ATOM_FEATURE_DIM = DGIN3_ATOM_FEATURE_DIM
EDGE_FEATURE_DIM = DGIN3_EDGE_FEATURE_DIM


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class BasicModelConfig:
    """
    Config for all graph neural network files.
    General model parameters
    """
    # name of the model - unique identifier that needs to be different for each model as it
    # is the name under which everything is safed.
    model_name: str = 'all_logs_dgin3_2'
    # batch size
    batch_size: int =15
    # if the model_name exists - should it override the existing model and all other things related
    # to this model?
    override_if_exists: bool = True
    # path to the project folder
    project_path:str = './'

    # do you want to retrain an existing model_name model?
    retrain_model: bool = False
    # if True, define the name and epoch of the model
    retrain_model_name: str = ''
    retrain_model_epoch: str = ''
    # do not change this - is done automatically
    retrain_model_weights_dir: str = project_path+'reports/model_weights/'+retrain_model_name+'/epoch_'+retrain_model_epoch+'/checkp_'+retrain_model_epoch

    # define where the train and test data (pickled graphs) are located (folder only)
    train_data_dir: str = project_path+'data/processed/lipo/pickled/train_dgin6_logd/'
    test_data_dir: str = project_path+'data/processed/lipo/pickled/test_dgin6_logd/'

    # are there two different directories (e.g. when using two datasets that have not benn merged)
    combined_dataset: bool = False
    # if there are two different directories - define the second directory for train/test (folders only)
    add_train_data_dir: str = project_path+'data/processed/lipo/pickled/train_dgin6_logs/'
    add_test_data_dir: str = project_path+'data/processed/lipo/pickled/test_dgin6_logs/'

    # do you want to test the model?
    test_model: bool = False
    # if yes, define the epoch you want to use.
    test_model_epoch: str = '609'
    # do not change this!
    test_model_weights_dir: str = project_path+'reports/model_weights/'+model_name+'/epoch_'+test_model_epoch+'/checkp_'+test_model_epoch

    # the directory to the log files.
    log_dir: str = project_path+'reports/logs/'+model_name+'.log'
    # the verbosity of the logging
    verbosity_level = logging.INFO

    # what kind of model do you want to train - can be either "GIN", "MPNN" or "DGIN"
    model_type: str = "DGIN" 
    
    ### do not change this! these paths are generated for the model under model_name
    plot_dir: str = project_path+'reports/figures/'+model_name+'/'
    tensorboard_log_dir: str = project_path+'reports/tensorboard/'+model_name+'/'
    config_log_dir: str = project_path+'reports/configs/'+model_name+'/'
    model_weights_dir: str = project_path+'reports/model_weights/'+model_name+'/'
    stats_log_dir: str = project_path+'reports/stats/'+model_name+'/'

@dataclass
class DGINConfig:
    """
    Specific submodel configs for the DGIN, MPNN or GIN model.
    """
    # initialize the DGIN or D-MPNN features with bias; the GIN's features are initialized without NN
    init_bias: bool = False
    
    ### DMPNN part:
    # include during aggregate part
    dropout_aggregate_dmpnn: bool = False
    layernorm_aggregate_dmpnn: bool = True
    dmpnn_passing_bias: bool = False
    # include during message passing part
    dropout_passing_dmpnn: bool = False
    layernorm_passing_dmpnn: bool = True
    # how many layers during the D-MPNN message iteration phase in the D-GIN or only D-MPNN model
    massge_iteration_dmpnn: int = 4

    ### GIN part:
    # include during aggregate part
    dropout_aggregate_gin: bool = False
    layernorm_aggregate_gin: bool = True
    gin_aggregate_bias: bool = False
    # include during passing part
    dropout_passing_gin: bool = False
    layernorm_passing_gin: bool = True
    # how many layers during the GIN message iteration phase in the D-GIN or only GIN model
    message_iterations_gin: int = 4

    # dropout used throughout the models
    dropout_rate: float = 0.15

    # do not change - input dimension of the models; is generated automatically
    input_size: int = (ATOM_FEATURE_DIM+EDGE_FEATURE_DIM) # combination of node feature len (33) and edge feature len (12)
    # can be changed - hidden feature size
    passing_hidden_size: int = 56
    # do not change - dimension of GIN input in the D-GIN or at the end of the GIN model during the aggregation phase
    input_size_gin: int = (ATOM_FEATURE_DIM+passing_hidden_size)

@dataclass
class Model1Config:
    """
    general model configurations.
    """
    # how the training and validation should be split; e.g. 0.90 are 90% training, 10% validation
    validation_split: float = 0.90
    # specific learning rate
    learning_rate: float = 0.004
    # 
    clip_rate: float = 0.6
    # optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate)
    # different losses - only MSE
    lipo_loss_mse = tf.keras.losses.mse
    logP_loss_mse = tf.keras.losses.mse
    logS_loss_mse = tf.keras.losses.mse
    
    # define how many epochs should be run
    epochs: int = 1600
    # define the number of epochs for each test run. 
    safe_after_epoch: int = 3
    # dropout rate for the general model - mainly the MLP for the different log predictions
    dropout_rate: float = 0.15 # the overall dropout rate of the readout functions
    # the seed to shuffle the training/validation dataset; For the same dataset, even when
    # combined_dataset is True, it is the same training/valiation instances
    train_data_seed: int = 0

    # hidden feature output size of the first layer in the MLP for the different log predictions.
    # can be changed
    hidden_readout_1: int = 32
    # hidden feature output size of the second layer in the MLP for the different log predictions.
    # can be changed
    hidden_readout_2: int = 14
    # activation function for the MLP for the different log predictions.
    # can be changed
    activation_func_readout = tf.nn.relu
    
    # define what property prediction should be included.
    # If set True but does not have that property, an exception is thrown BUT the model
    # continuous 
    include_logD: bool = True
    include_logS: bool = True
    include_logP: bool = True

    # define the starting threshold for the RMSE of the model. When the comnbined RMSE
    # is below this threshold, the model weights are being safed and a new threshold
    # is set. It only serves as a starting threshold so that not too many models
    # are being safed. Depends on how many log endpoints are being taken into
    # consideration - as three endpoints have a higher combined RMSE as only one 
    # endpoint.
    best_evaluation_threshold: float = 1.45 #was introduced on the 25.03.2021/ 
                                            # 2.45 for all_logs
                                            # 0.70 logP
                                            # 0.75 logD
                                            # 1.00 logS
                                            # 1.75 logSD
                                            # 1.70 logSP
                                            # 1.45 logDP
    # do you want to use RMSE instead of MSE for the different losses?
    # True if yes, False if not
    use_rmse: bool = True

@dataclass
class Config():
    """
    Dataclass for all other configs.
    """
    basic_model_config: BasicModelConfig
    model1_config: Model1Config
    d_gin_config: DGINConfig
    # define the model type you want to use.
    # currently available: 'model10' and 'model11'
    model: str = 'model10'
