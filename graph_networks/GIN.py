import tensorflow as tf
import numpy as np
import copy
from graphnets.utilities_nn import CustomDropout as dropout

class GIN(tf.keras.Model):
    '''
    This class is an implementation of the GIN-E model (Edge extention of Xu et al. 2019).

    '''
    
    def __init__(self,config,return_hv=False):
        super(GIN, self).__init__(name="GIN")
        
        self.config = config
        self.return_hv = return_hv
        if self.config.layernorm_passing_gin:
            self.layernorm_passing_gin = tf.keras.layers.LayerNormalization(epsilon=1e-6,axis=1)
        if self.config.dropout_passing_gin:
            self.dropout_passing_gin = tf.keras.layers.Dropout(self.config.dropout_rate)

        if self.config.layernorm_aggregate_gin:
            self.layernorm_aggregate_gin = tf.keras.layers.LayerNormalization(epsilon=1e-6,axis=1)
        if self.config.dropout_aggregate_gin:
            self.dropout_aggregate_gin = tf.keras.layers.Dropout(self.config.dropout_rate)

        self.eps = tf.Variable(tf.zeros((self.config.input_size_gin,)),trainable=True)
        self.massge_iteration_gin = config.message_iterations_gin
        self.gin_aggregate = list()
        for t in range(0,self.massge_iteration_gin):
            self.gin_aggregate.append(tf.keras.layers.Dense(self.config.input_size_gin,
            activation=None,input_shape=(self.config.input_size_gin,),
            name="gin_aggregate"+str(t),use_bias=self.config.gin_aggregate_bias))

    def call(self,instance,train=True):
        edge_aligned_node_features = tf.convert_to_tensor(instance.edge_aligned_node_features,dtype=tf.dtypes.float32)
        dir_edge_features = tf.convert_to_tensor(instance.dir_edge_features,dtype=tf.dtypes.float32)
        node_features = tf.convert_to_tensor(instance.node_features,dtype=tf.dtypes.float32)
        # initialize
        # aggregate info and combine
        h_0_v = node_features
        h_v = 0
        # massage passing gin
        for t in range(0,self.massge_iteration_gin):
            if t == 0:
                h_v = tf.matmul(tf.convert_to_tensor(instance.adj_matrix,
                    dtype=tf.dtypes.float32),h_0_v)
            else:
                h_v = tf.matmul(tf.convert_to_tensor(instance.adj_matrix,
                    dtype=tf.dtypes.float32),h_v)
            if self.config.layernorm_passing_gin:
                h_v = self.layernorm_passing_gin(h_v)
            if self.config.dropout_passing_gin:
                h_v = self.dropout_passing_gin(h_v,training=train)
            h_v =self.gin_aggregate[t]((1+self.eps)*h_0_v+h_v)
        if self.return_hv:
            return h_v
            
        h = tf.reduce_sum(h_v,axis=0)
        h = tf.reshape(h,[1,self.config.input_size_gin])
        if self.config.layernorm_aggregate_gin:
            h = self.layernorm_aggregate_gin(h)
        if self.config.dropout_aggregate_gin:
            h = self.dropout_aggregate_gin(h,training=train)

        return h