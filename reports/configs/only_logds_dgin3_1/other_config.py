from dataclasses import dataclass, field
from typing import List

import tensorflow as tf
from graph_networks.utilities import * 
import logging
import os

ATOM_FEATURE_DIM = DGIN3_ATOM_FEATURE_DIM
EDGE_FEATURE_DIM = DGIN3_EDGE_FEATURE_DIM

@dataclass
class BasicModelConfig:
    """
    Config for model1/2/3 run file.
    General model parameters
    """
    model_name: str = 'only_logds_dgin3_1' # without h_w in DGIN gin part - added h_v_0 instead
                                    # whole train/eval split - no more double split within train data set
                                    # random train/test split in get_data_sd - only change overall_seed
                                    # CHANGES dgin3 10.02.2021:
                                    # *added new bondFeaturesDGIN2 and atomFeaturesDGIN2; DGIN2_ATOM_FEATURE_DIM; DGIN2_EDGE_FEATURE_DIM
                                    # *from project_path+'data/processed/lipo/pickled/train_frags3/' to project_path+'data/processed/lipo/pickled/test_frags3/'
                                    # CHANGES dgin3 16.02.2021:
                                    # *added new bondFeaturesDGIN3 and atomFeaturesDGIN3; DGIN3_ATOM_FEATURE_DIM; DGIN3_EDGE_FEATURE_DIM
                                    # *from project_path+'data/processed/lipo/pickled/train_frags_dgin3/' to project_path+'data/processed/lipo/pickled/test_frags_dgin3/'
                                    # CHANGES dgin4 16.02.2021:
                                    # *added add_species bool in model1 config - previously not there; for dgin2 featurization adds the species type after the dgin 
                                    # encoding before logD prediction
                                    # test_frags_dgin4 was added for species inclusion in model2 call()
    batch_size: int =15
    override_if_exists: bool = True

    overall_seed: int = 2
    
    # path to the project folder 
    project_path:str = "./" 

    retrain_model: bool = False
    retrain_model_name: str = ''
    retrain_model_epoch: str = ''
    retrain_model_weights_dir: str = project_path+'reports/model_weights/'+retrain_model_name+'/epoch_'+retrain_model_epoch+'/checkp_'+retrain_model_epoch

    train_data_dir: str = project_path+'data/processed/lipo/pickled/train_dgin3_logd/'
    test_data_dir: str = project_path+'data/processed/lipo/pickled/test_dgin3_logd/'

    combined_dataset: bool = True

    add_train_data_dir: str = project_path+'data/processed/lipo/pickled/train_dgin3_logs/'
    add_test_data_dir: str = project_path+'data/processed/lipo/pickled/test_dgin3_logs/'

    test_model: bool = False
    test_model_epoch: str = '887'

    # define the number or test runs for the CI. 
    # the mean and std of the RMSE and r^2 of the combined runs are taken as the output. 
    test_n_times: int = 1 
    # do you want to test the model with consensus mode? 
    # if yes, a defined ML model will be included in the consensus predictions during the testing. 
    consensus: bool = False 
    # include dropout during testing?
    include_dropout: bool = False
    test_model_weights_dir: str = project_path+'reports/model_weights/'+model_name+'/epoch_'+test_model_epoch+'/checkp_'+test_model_epoch

    # To save the prediction values for each property set to True 
    # When this flag is True - the whole test dataset is taken an test_n_times is set to zero! 
    save_predictions: bool = False 
    # define the folder where you want to save the predictions. 
    # For each property, a file is created under the property name ("./logd.txt","./logs.txt","./logp.txt","./others.txt") 
    test_prediction_output_folder: str = project_path+"reports/predictions/"+model_name+"/" 
    encode_hidden: bool = False

    log_dir: str = project_path+'reports/logs/'+model_name+'.log' 
    verbosity_level = logging.INFO
    
    plot_dir: str = project_path+'reports/figures/'+model_name+'/'
    tensorboard_log_dir: str = project_path+'reports/tensorboard/'+model_name+'/'
    config_log_dir: str = project_path+'reports/configs/'+model_name+'/'
    model_weights_dir: str = project_path+'reports/model_weights/'+model_name+'/'
    stats_log_dir: str = project_path+'reports/stats/'+model_name+'/'

@dataclass
class DGINConfig:
    """
    Config for direcpted-mpnn class.
    """
    dropout_aggregate_dmpnn: bool = False
    layernorm_aggregate_dmpnn: bool = True
    dropout_passing_dmpnn: bool = False
    layernorm_passing_dmpnn: bool = True

    dropout_aggregate_gin: bool = False
    layernorm_aggregate_gin: bool = True
    dropout_passing_gin: bool = False
    layernorm_passing_gin: bool = True

    gin_aggregate_bias: bool = False
    dmpnn_passing_bias: bool = False
    init_bias: bool = False

    massge_iteration_dmpnn: int = 4
    message_iterations_gin: int = 4
    dropout_rate: float = 0.15
    input_size: int = (ATOM_FEATURE_DIM+EDGE_FEATURE_DIM) # combination of node feature len (33) and edge feature len (12)
    passing_hidden_size: int = 56 # this can be changed
    input_size_gin: int = (ATOM_FEATURE_DIM+passing_hidden_size)

    return_hv: bool = True # model3 parameter

@dataclass
class Model1Config:
    """
    Config model1 class - no subclass configs are defined here.
    """
    validation_split: float = 0.90
    learning_rate: float = 0.004
    clip_rate: float = 0.6
    optimizer = tf.keras.optimizers.Adam(learning_rate)
    lipo_loss_mse = tf.keras.losses.mse
    lipo_loss_mae = tf.keras.losses.mae
    logP_loss_mse = tf.keras.losses.mse
    logS_loss_mse = tf.keras.losses.mse
    other_loss_mse = tf.keras.losses.mse 
    mw_loss_mse = tf.keras.losses.mse
    metric = tf.keras.losses.mae
    epochs: int = 1600
    # define the number of epochs for each test run.  
    save_after_epoch: int = 3 
    # dropout rate for the general model - mainly the MLP for the different log predictions 
    dropout_rate: float = 0.15 # the overall dropout rate of the readout functions 
    # the seed to shuffle the training/validation dataset; For the same dataset, even when 
    # combined_dataset is True, it is the same training/valiation instances 
    train_data_seed: int = 0 
    dropout_rate: float = 0.15 # the overall dropout rate of the readout functions
    train_data_seed: int = 0

    hidden_readout_1: int = 32
    hidden_readout_2: int = 14
    activation_func_readout = tf.nn.relu
    
    include_logD: bool = True
    include_logS: bool = True
    include_logP: bool = False

    include_other: bool = False 
    include_mw: bool = False
    include_rot_bond: bool = False
    include_HBA: bool = False
    include_HBD: bool = False

    # define the starting threshold for the RMSE of the model. When the comnbined RMSE 
    # is below this threshold, the model weights are being safed and a new threshold 
    # is set. It only serves as a starting threshold so that not too many models 
    # are being safed. Depends on how many log endpoints are being taken into 
    # consideration - as three endpoints have a higher combined RMSE as only one 
    # endpoint. 
    best_evaluation_threshold: float = 2.45 #was introduced on the 25.03.2021/ 

    # define the individual thresholds. If one model is better, the corresponding 
    # model weights are being saved. 
    best_evaluation_threshold_logd: float = 1.85 
    best_evaluation_threshold_logp: float = 1.65 
    best_evaluation_threshold_logs: float = 2.15 
    best_evaluation_threshold_other: float = 2.15 
                                            # 2.45 for all_logs
                                            # 0.70 logP
                                            # 0.75 logD
                                            # 1.00 logS
                                            # 1.75 logSD
                                            # 1.70 logSP
                                            # 1.45 logDP

    include_fragment_conv: bool = False # was introduced on the 4.12.2020

    use_rmse: bool = True # uses RMSE instead of MSE for only lipo_loss
    shuffle_inside: bool = True # reshuffles the train/valid test seach in each epoch (generalizes)

    add_species: bool = False # 16.02 introduction; previously not there; for dgin3 adds the species type after the dgin encoding before logD prediction

@dataclass
class FrACConfig:
    """
    Config fragment aggregation class - no subclass configs are defined here.
    """
    input_size_gin: int = 28
    layernorm_aggregate: bool = True
    reduce_mean: bool = True # when false -> reduce_sum

@dataclass 
class MLConfig: 
    """ 
    Configs for the ML algorithm 
    """ 
    # which algorithm do you want to use for the consensus? 
    # possibilities are: "SVM", "RF", "KNN" or "LR" - all are regression models! 
        # SVM: Support Vector Machine; RF: Random Forest, KNN: K-Nearest Neigbors; LR: Linear Regression;
    algorithm: str = "SVM" 
    # which fingerprint to use - possibilities are: "ECFP" or "MACCS" 
    fp_types: str = "ECFP" 
    # If 'ECFP' fingerprint is used, define the number of bits - maximum is 2048! 
    n_bits: int = 2048 
    # If "ECFP" fingerprint is used, define the radius 
    radius: int = 4 
    # define if descriptors should be included into the non-GNN molecular representation 
    include_descriptors: bool = True 
    # define if the descriptors should be standardizedby scaling and centering (Sklearn) 
    standardize: bool = True 

@dataclass
class Config():
    """
    Overall config class for model2 and run file.
    Includes all submodels config
    """
    basic_model_config: BasicModelConfig
    model1_config: Model1Config
    d_gin_config: DGINConfig
    frag_acc_config: FrACConfig

    ml_config: MLConfig 
    model: str = 'model10'