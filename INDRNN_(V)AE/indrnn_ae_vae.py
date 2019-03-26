# -*- coding: utf-8 -*-
"""
A simple Implementation of INDRNN_(V)AE based algorithm 
for both Anomaly(Novelty) Detection  in Multivariate Time Series;
We also persent a health-judge mechanism for accessing the statement of 
the input Multivariate Time Series, which might be useful in machine maintenance;


A special note between LSTM_VAE and INDRNN_(V)AE is that INDRNN_(V)AE
can be adopted by high-frequency scenarios (industry sensors,for example).

Author: Schindler Liang

Reference:
    https://github.com/twairball/keras_lstm_vae
    https://github.com/batzner/indrnn
"""

import numpy as np
import tensorflow as tf
from tensorflow.nn.rnn_cell import MultiRNNCell
from ind_rnn_cell import IndRNNCell

xavier_init = tf.contrib.layers.xavier_initializer(seed=2019)
zero_init =  tf.zeros_initializer()

def _INDRNNCells(unit_list,time_steps):
    recurrent_max = pow(2, 1 / time_steps)
    return MultiRNNCell([IndRNNCell(unit,recurrent_max_abs=recurrent_max) 
                         for unit in unit_list],state_is_tuple=True)

class Data_Hanlder:
    def __init__(self,train_file):
        self.train_data = np.load(train_file)
        
    def fetch_data(self,batch_size):
        indices = np.random.choice(self.train_data.shape[0],batch_size)
        return self.train_data[indices]
    
class INDRNN_VAE(object):
    def __init__(self,train_file,
                 z_dim=10,
                 encoder_layers=2,
                 decode_layers=2,
                 outlier_fraction=0.01
                 ):
        
        self.outlier_fraction = outlier_fraction
        self.data_source = Data_Hanlder(train_file)
        self.n_hidden = 16
        self.batch_size = 128
        self.learning_rate = 0.0005
        self.train_iters = 7000
        self.encoder_layers = encoder_layers
        self.decode_layers = decode_layers
        
        self.time_steps = self.data_source.train_data.shape[1]        
        self.input_dim = self.data_source.train_data.shape[2]
        self.z_dim = z_dim
    
        self.anomaly_score = 0
        self.sess = tf.Session()
        self._build_network()
        self.sess.run(tf.global_variables_initializer())
        
    def _build_network(self):
        with tf.variable_scope('ph'):
            self.X = tf.placeholder(tf.float32,shape=[None,self.time_steps,self.input_dim],name='input_X')
                        
        with tf.variable_scope('encoder',initializer=xavier_init):
            with tf.variable_scope('AE'):
                ae_fw_lstm_cells = _INDRNNCells([self.n_hidden]*self.encoder_layers,self.time_steps)
                ae_bw_lstm_cells = _INDRNNCells([self.n_hidden]*self.encoder_layers,self.time_steps)
                (ae_fw_outputs,ae_bw_outputs),_ = tf.nn.bidirectional_dynamic_rnn(
                                                        ae_fw_lstm_cells,
                                                        ae_bw_lstm_cells, 
                                                        self.X, dtype=tf.float32)
                ae_outputs = tf.add(ae_fw_outputs,ae_bw_outputs)
                
                
                
            with tf.variable_scope('lat_Z'):
                z_fw_lstm_cells = _INDRNNCells([self.n_hidden]*self.encoder_layers,
                                               self.time_steps)
                z_bw_lstm_cells = _INDRNNCells([self.n_hidden]*self.encoder_layers,
                                               self.time_steps)
                (z_fw_outputs,z_bw_outputs),_ = tf.nn.bidirectional_dynamic_rnn(
                                                            z_fw_lstm_cells,
                                                            z_bw_lstm_cells, 
                                                            self.X, dtype=tf.float32)
                z_outputs = tf.reduce_mean( (z_fw_outputs+z_bw_outputs),axis=1 )
                
                mu_outputs = tf.layers.dense(z_outputs,self.z_dim,activation=tf.nn.tanh)               
                log_sigma_outputs = tf.layers.dense(z_outputs,self.z_dim)               
        
                sample_Z =  mu_outputs + tf.exp(log_sigma_outputs/2) * tf.random_normal(
                                                        tf.shape(mu_outputs),
                                                        0,1,dtype=tf.float32)
        
        
        with tf.variable_scope('decoder'):
            sample_Z = tf.expand_dims(sample_Z,axis=1)
            sample_Z = tf.tile(sample_Z,[1,self.time_steps,1])
            decoder_input = tf.concat([ae_outputs,sample_Z],axis=-1)
            
            recons_fw_lstm_cells = _INDRNNCells([self.n_hidden]*self.decode_layers + [self.input_dim],
                                                 self.time_steps)
            recons_bw_lstm_cells = _INDRNNCells([self.n_hidden]*self.decode_layers + [self.input_dim],
                                                 self.time_steps)
            
            (recons_fw_outputs,recons_bw_outputs),_ = tf.nn.bidirectional_dynamic_rnn( 
                                                                recons_fw_lstm_cells,
                                                                recons_bw_lstm_cells, 
                                                                decoder_input, dtype=tf.float32)           
            self.recons_X = tf.add(recons_fw_outputs,recons_bw_outputs)
 
        with tf.variable_scope('loss'):
            reduce_dims = np.arange(1,tf.keras.backend.ndim(self.X))
            recons_loss = tf.losses.mean_squared_error(self.X, self.recons_X)
            kl_loss = - 0.5 * tf.reduce_mean(1 + log_sigma_outputs - tf.square(mu_outputs) - tf.exp(log_sigma_outputs))
            self.opt_loss = recons_loss + kl_loss
            self.all_losses = tf.reduce_sum(tf.square(self.X - self.recons_X),reduction_indices=reduce_dims)

        with tf.variable_scope('train'):
            self.uion_train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.opt_loss)
            
            
    def train(self):
        for i in range(self.train_iters):            
            this_X = self.data_source.fetch_data(self.batch_size)
            self.sess.run([self.uion_train_op],feed_dict={
                    self.X: this_X
                    })
            if i % 200 ==0:
                mse_loss = self.sess.run([self.opt_loss],feed_dict={
                    self.X: self.data_source.train_data
                    })
                print('epoch {}: with loss: {}'.format(i,mse_loss))
        self._arange_score(self.data_source.train_data)
    
    def _arange_score(self,input_data):
        all_losses = self.sess.run(self.all_losses,feed_dict={
                    self.X: input_data
                    })
        self.sorted_loss = np.sort(all_losses).ravel()

        self.anomaly_score = np.percentile(self.sorted_loss,(1-self.outlier_fraction)*100)
    
       
    def judge_health(self,test):

        all_losses = self.sess.run(self.all_losses,feed_dict={
                    self.X: test
                    }).ravel()       
        percentile_95 = self.sorted_loss[int(self.sorted_loss.shape[0]*0.95)]
        value_gap = self.sorted_loss[-1] - percentile_95
        def _get_health(loss):
            min_index = np.argmin(np.abs(self.sorted_loss-loss))
            if min_index < self.sorted_loss.shape[0] - 1:
                minus_ratio = min_index / self.sorted_loss.shape[0]               
            else:
                exceed_loss = loss - self.sorted_loss[-1]
                minus_ratio = exceed_loss / value_gap * 0.05 + 1               
            return 100.0 - 40 * minus_ratio
        all_health = list(map(lambda x:_get_health(x),all_losses))        
        return all_health

    def judge_anomaly(self,test):
        all_losses = self.sess.run(self.all_losses,feed_dict={
                    self.X: test
                    }).ravel()
        

        judge_label = list( map(lambda x: -1 if x>self.anomaly_score else 1,all_losses)   )
        
        return judge_label

    
indrnn_ae = INDRNN_VAE(train_file='dataset/train.npy',z_dim=10,outlier_fraction=0.04)    
indrnn_ae.train()  

test = np.load('dataset/test.npy')
z1 = indrnn_ae.judge_health(test)
z2 = indrnn_ae.judge_anomaly(test)

import matplotlib.pyplot as plt
plt.plot(z1)
