import logging
log = logging.getLogger(__name__)

import numpy as np
import os
import datetime
import time
import pandas as pd
import random
import sys
import copy

import tensorflow as tf
from tensorflow.keras.layers import Dense as dense
from tensorboard.plugins import projector
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

class Model10(tf.keras.Model):
    """[summary]
    This model is used to train, validate and test logD logS and logP endpoints by deploying
    either D-MPNN, GIN or D-GIN architectures.
    It is an extention of the base model class.
    """
    def __init__(self,config):
        super(Model11, self).__init__(name="Model-10")