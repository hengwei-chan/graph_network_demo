#!/usr/bin/env python

# =============================================================================
# MODULE DOCSTRING
# =============================================================================

"""

"""

# =============================================================================
# GLOBAL IMPORTS
# =============================================================================

import logging
log = logging.getLogger(__name__)
logging.getLogger('tensorflow').setLevel(logging.ERROR)

import numpy as np
import os
import datetime
import time
import pandas as pd
import random
import sys
import copy

from sklearn.metrics import r2_score

import tensorflow as tf
from tensorflow.keras.layers import Dense as dense
from tensorboard.plugins import projector
from sklearn.metrics import mean_squared_error

from graph_networks.DGIN import DGIN
from graph_networks.GIN import GIN
from graph_networks.DMPNN import DMPNN

import os

class BaseModel(tf.keras.Model):
    """[summary]
    Super model interface for all other gnn models.
    """
    def __init__(self,config):
        super(BaseModel, self).__init__(name="BaseModel")

        self.model_name = config.basic_model_config.model_name
        self.batch_size = config.basic_model_config.batch_size
        self.plot_dir =config.basic_model_config.plot_dir
        self.safe_model_dir = config.basic_model_config
        self.stats_log_dir = config.basic_model_config.stats_log_dir
        self.model_weights_dir = config.basic_model_config.model_weights_dir
        self.tensorboard_log_dir = config.basic_model_config.tensorboard_log_dir
        self.project_path = config.basic_model_config.project_path
        self.test_prediction_output_folder = config.basic_model_config.test_prediction_output_folder
        
        self.combined_dataset = config.basic_model_config.combined_dataset
        self.save_predictions = config.basic_model_config.save_predictions

        self.config = config.model1_config

        if config.basic_model_config.model_type == 'GIN':
            self.dgin = GIN(config.d_gin_config)
        #     if not '_gin' in config.basic_model_config.model_name:
        #         logging.error("Misleading model_name ("+config.basic_model_config.model_name +") - does not fit model_type: "+config.basic_model_config.model_type)
        #         raise Exception("Misleading model_name - does not fit model_type: "+config.basic_model_config.model_type)
        elif config.basic_model_config.model_type == 'DMPNN':
            self.dgin = DMPNN(config.d_gin_config)
        #     if not 'dmpnn' in config.basic_model_config.model_name:
        #         logging.error("Misleading model_name ("+config.basic_model_config.model_name +") - does not fit model_type: "+config.basic_model_config.model_type)
        #         raise Exception("Misleading model_name - does not fit model_type: "+config.basic_model_config.model_type)
        elif config.basic_model_config.model_type == 'DGIN':
            self.dgin = DGIN(config.d_gin_config)
        #     if not 'dgin' in config.basic_model_config.model_name:
        #         logging.error("Misleading model_name ("+config.basic_model_config.model_name +") - does not fit model_type: "+config.basic_model_config.model_type)
        #         raise Exception("Misleading model_nameb - does not fit model_type: "+config.basic_model_config.model_type)
        else:
            logging.error("No proper model_type added in the basic_model_config: "+config.basic_model_config.model_type)
            raise Exception("No proper model_type added in the basic_model_config: "+config.basic_model_config.model_type)

        self.best_validation_overall = self.config.best_evaluation_threshold
        self.best_validation_logd = self.config.best_evaluation_threshold_logd
        self.best_validation_logs = self.config.best_evaluation_threshold_logs
        self.best_validation_logp = self.config.best_evaluation_threshold_logp
        self.best_evaluation_threshold_other = self.config.best_evaluation_threshold_other

        if self.config.include_logD:
            self.lipo_loss_mse = self.config.lipo_loss_mse
            self.logD = tf.keras.Sequential([
                tf.keras.layers.Dropout(self.config.dropout_rate),
                tf.keras.layers.Dense(self.config.hidden_readout_1, activation=self.config.activation_func_readout, 
                        input_shape=(config.d_gin_config.input_size_gin,),
                        name="first_layer_readout_logD",
                        kernel_initializer=tf.keras.initializers.he_normal()),
                tf.keras.layers.Dropout(self.config.dropout_rate),
                tf.keras.layers.Dense(self.config.hidden_readout_2, activation=self.config.activation_func_readout, 
                        input_shape=(self.config.hidden_readout_1,),
                        name="sec_layer_readout_logD",
                        kernel_initializer=tf.keras.initializers.he_normal()),
                tf.keras.layers.Dropout(self.config.dropout_rate),
                tf.keras.layers.Dense(1,activation=None,kernel_initializer=tf.keras.initializers.he_normal(),
                        input_shape=(self.config.hidden_readout_2,))
                ], name="readout_logD")

        if self.config.include_logP:
            self.logP_loss_mse = self.config.logP_loss_mse
            self.logP = tf.keras.Sequential([
                tf.keras.layers.Dropout(self.config.dropout_rate),
                tf.keras.layers.Dense(self.config.hidden_readout_1, activation=self.config.activation_func_readout, 
                        input_shape=(config.d_gin_config.input_size_gin,),
                        name="first_layer_readout_logP",
                        kernel_initializer=tf.keras.initializers.he_normal()),
                tf.keras.layers.Dropout(self.config.dropout_rate),
                tf.keras.layers.Dense(self.config.hidden_readout_2, activation=self.config.activation_func_readout, 
                        input_shape=(self.config.hidden_readout_1,),
                        name="sec_layer_readout_logP",
                        kernel_initializer=tf.keras.initializers.he_normal()),
                tf.keras.layers.Dropout(self.config.dropout_rate),
                tf.keras.layers.Dense(1,activation=None,kernel_initializer=tf.keras.initializers.he_normal(),
                        input_shape=(self.config.hidden_readout_2,))
                ], name="readout_x_logP")

        if self.config.include_logS:
            self.logS_loss_mse = self.config.logS_loss_mse
            self.logS = tf.keras.Sequential([
                tf.keras.layers.Dropout(self.config.dropout_rate),
                tf.keras.layers.Dense(self.config.hidden_readout_1, activation=self.config.activation_func_readout, 
                        input_shape=(config.d_gin_config.input_size_gin,),
                        name="first_layer_readout_logS",
                        kernel_initializer=tf.keras.initializers.he_normal()),
                tf.keras.layers.Dropout(self.config.dropout_rate),
                tf.keras.layers.Dense(self.config.hidden_readout_2, activation=self.config.activation_func_readout, 
                        input_shape=(self.config.hidden_readout_1,),
                        name="sec_layer_readout_logS",
                        kernel_initializer=tf.keras.initializers.he_normal()),
                tf.keras.layers.Dropout(self.config.dropout_rate),
                tf.keras.layers.Dense(1,activation=None,kernel_initializer=tf.keras.initializers.he_normal(),
                        input_shape=(self.config.hidden_readout_2,))
                ], name="readout_logS")

        if self.config.include_other:
            self.other_loss_mse = self.config.other_loss_mse
            self.other = tf.keras.Sequential([
                tf.keras.layers.Dropout(self.config.dropout_rate),
                tf.keras.layers.Dense(self.config.hidden_readout_1, activation=self.config.activation_func_readout, 
                        input_shape=(config.d_gin_config.input_size_gin,),
                        name="first_layer_readout_other",
                        kernel_initializer=tf.keras.initializers.he_normal()),
                tf.keras.layers.Dropout(self.config.dropout_rate),
                tf.keras.layers.Dense(self.config.hidden_readout_2, activation=self.config.activation_func_readout, 
                        input_shape=(self.config.hidden_readout_1,),
                        name="sec_layer_readout_other",
                        kernel_initializer=tf.keras.initializers.he_normal()),
                tf.keras.layers.Dropout(self.config.dropout_rate),
                tf.keras.layers.Dense(1,activation=None,kernel_initializer=tf.keras.initializers.he_normal(),
                        input_shape=(self.config.hidden_readout_2,))
                ], name="readout_other")
        
    def run(self):
        '''
        This is the main method that runs the model by calling train, test or evaluate methods.
        In each of these methods, the model is called and a prediction for each instance is generated.
        '''
        
        return

    def call(self):
        '''
        This method gets one instance and generates a prediction.
        '''
        return

    def train(self, batch,overall_iterations,train_summary_writer):
        '''
        Trains the compiled model by calling the call method.
        '''

        return

    def evaluate(self):
        '''
        evaluates the model by using the evaluation data by calling the call method.
        '''
        return


    def test(self, test_data,epoch,test_losses=None,save_weights_flag=True,include_dropout=False,predict_n_times=1,
            simply_testing=True):
        '''
        tests the trained model by using the test data by calling the call method.

        '''

        return

    def test_n_times(self, test_data,epoch,test_losses=None,save_weights_flag=True,include_dropout=False,predict_n_times=1,
            simply_testing=True,also_evaluate=False,eval_data=None):
        '''
        tests the trained model by using the test data by calling the call method.
        difference to the test method is that here we use convidence intervalls.
        '''

        return