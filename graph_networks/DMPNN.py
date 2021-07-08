import tensorflow as tf
import numpy as np
import copy
from graph_networks.utilities import CustomDropout as dropout

class DMPNN(tf.keras.Model):
    '''
    This class is an implementation of the directed massage passing (Yang et al. 2019).

    '''
    
    def __init__(self,config,return_hv=False):
        super(DMPNN, self).__init__(name="DMPNN")
        
        self.config = config
        self.return_hv = return_hv
        # D-MPNN
        self.init = tf.keras.layers.Dense(self.config.passing_hidden_size,activation='relu',
                input_shape=(self.config.input_size,),name="init",
                use_bias=self.config.init_bias)
        self.passing = list()
        self.massge_iteration_dmpnn = self.config.massge_iteration_dmpnn

        self.aggregate = tf.keras.layers.Dense(self.config.passing_hidden_size,
                activation='relu',input_shape=(self.config.passing_hidden_size,),
                name="aggregate",use_bias=self.config.dmpnn_passing_bias)

        for t in range(0,self.massge_iteration_dmpnn):
            self.passing.append(tf.keras.layers.Dense(self.config.passing_hidden_size,
                activation=None,input_shape=(self.config.passing_hidden_size,),
                name="message"+str(t),use_bias=self.config.dmpnn_passing_bias))
        
        if self.config.layernorm_passing_dmpnn:
            self.layernorm_passing_dmpnn = tf.keras.layers.LayerNormalization(epsilon=1e-6,axis=1)
        if self.config.dropout_passing_dmpnn:
            # self.dropout_passing_dmpnn = dropout(self.config.dropout_rate)
            self.dropout_passing_dmpnn = tf.keras.layers.Dropout(self.config.dropout_rate)
        
        if self.config.layernorm_aggregate_dmpnn:
            self.layernorm_aggregate_dmpnn = tf.keras.layers.LayerNormalization(epsilon=1e-6,axis=1)
        if self.config.dropout_aggregate_dmpnn:
            # self.dropout_aggregate_dmpnn = dropout(self.config.dropout_rate)
            self.dropout_aggregate_dmpnn = tf.keras.layers.Dropout(self.config.dropout_rate)


    def call(self,instance,train=True):
        edge_aligned_node_features = tf.convert_to_tensor(instance.edge_aligned_node_features,dtype=tf.dtypes.float32)
        dir_edge_features = tf.convert_to_tensor(instance.dir_edge_features,dtype=tf.dtypes.float32)
        node_features = tf.convert_to_tensor(instance.node_features,dtype=tf.dtypes.float32)
        # initialize
        h_0_vw = self.init(tf.concat([edge_aligned_node_features,
            dir_edge_features],axis=1))
        # massage passing d-mpnn
        m_t_vw = 0
        h_t_vw = tf.identity(h_0_vw)
        for t in range(0,self.massge_iteration_dmpnn):
            if t == 0:
                m_t_vw = tf.matmul(tf.convert_to_tensor(instance.adj_matrix_edges_wo,
                    dtype=tf.dtypes.float32),h_0_vw)
            else:
                m_t_vw = tf.matmul(tf.convert_to_tensor(instance.adj_matrix_edges_wo,
                    dtype=tf.dtypes.float32),h_t_vw)
            if self.config.layernorm_passing_dmpnn:
                m_t_vw = self.layernorm_passing_dmpnn(m_t_vw)
            if self.config.dropout_passing_dmpnn:
                m_t_vw = self.dropout_passing_dmpnn(m_t_vw,training=train)
            h_t_vw = tf.nn.relu(tf.math.add(h_0_vw,self.passing[t](m_t_vw)))

        m_v = tf.matmul(tf.convert_to_tensor(instance.atm_dir_edge_adj_matrix,
            dtype=tf.dtypes.float32),h_t_vw)
        if self.config.layernorm_aggregate_dmpnn:
            m_v = self.layernorm_aggregate_dmpnn(m_v)
        if self.config.dropout_aggregate_dmpnn:
            m_v = self.dropout_aggregate_dmpnn(m_v,training=train)
        
        h_v =self.aggregate(tf.concat([node_features,m_v],axis=1))
        
        if self.return_hv:
            return h_v
        
        if self.config.layernorm_aggregate_dmpnn:
            h_v = self.layernorm_aggregate_dmpnn(h_v)
        if self.config.dropout_aggregate_dmpnn:
            h_v = self.dropout_aggregate_dmpnn(h_v,training=train)

        h = tf.reduce_sum(h_v,axis=0)
        h = tf.reshape(h,[1,self.config.passing_hidden_size])
            
        return h