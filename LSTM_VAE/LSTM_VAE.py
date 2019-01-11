# -*- coding: utf-8 -*-
"""
One simple Implementation of LSTM_VAE based algorithm for Anomaly Detection in Multivariate Time Series;

Author: Schindler Liang

Reference:
    https://www.researchgate.net/publication/304758073_LSTM-based_Encoder-Decoder_for_Multi-sensor_Anomaly_Detection
    https://github.com/twairball/keras_lstm_vae
    https://arxiv.org/pdf/1711.00614.pdf    
"""
import numpy as np
import tensorflow as tf
from tensorflow.nn.rnn_cell import MultiRNNCell, LSTMCell
from utils import Data_Hanlder


def lrelu(x, leak=0.2, name='lrelu'):
	return tf.maximum(x, leak*x)


def _LSTMCells(unit_list,act_fn_list):
    return MultiRNNCell([LSTMCell(unit,                         
                         activation=act_fn) 
                         for unit,act_fn in zip(unit_list,act_fn_list )])
    
class LSTM_VAE(object):
    def __init__(self,dataset_name,columns,z_dim,time_steps,outlier_fraction):
        self.outlier_fraction = outlier_fraction
        self.data_source = Data_Hanlder(dataset_name,columns,time_steps)
        self.n_hidden = 16
        self.batch_size = 128
        self.learning_rate = 0.0005
        self.train_iters = 4000
        
        self.input_dim = len(columns)
        self.z_dim = z_dim
        self.time_steps = time_steps
    
        self.pointer = 0 
        self.anomaly_score = 0
        self.sess = tf.Session()
        self._build_network()
        self.sess.run(tf.global_variables_initializer())
        
    def _build_network(self):
        with tf.variable_scope('ph'):
            self.X = tf.placeholder(tf.float32,shape=[None,self.time_steps,self.input_dim],name='input_X')
        
        with tf.variable_scope('encoder'):
            with tf.variable_scope('lat_mu'):
                mu_fw_lstm_cells = _LSTMCells([self.z_dim],[lrelu])
                mu_bw_lstm_cells = _LSTMCells([self.z_dim],[lrelu])

                (mu_fw_outputs,mu_fw_outputs),_ = tf.nn.bidirectional_dynamic_rnn(
                                                        mu_fw_lstm_cells,
                                                        mu_bw_lstm_cells, 
                                                        self.X, dtype=tf.float32)
                mu_outputs = tf.add(mu_fw_outputs,mu_fw_outputs)
                
            with tf.variable_scope('lat_sigma'):
                sigma_fw_lstm_cells = _LSTMCells([self.z_dim],[tf.nn.softplus])
                sigma_bw_lstm_cells = _LSTMCells([self.z_dim],[tf.nn.softplus])
                (sigma_fw_outputs,sigma_bw_outputs),_ = tf.nn.bidirectional_dynamic_rnn(
                                                            sigma_fw_lstm_cells,
                                                            sigma_bw_lstm_cells, 
                                                            self.X, dtype=tf.float32)
                sigma_outputs = tf.add(sigma_fw_outputs,sigma_bw_outputs)                 
                sample_Z =  mu_outputs + sigma_outputs * tf.random_normal(
                                                        tf.shape(mu_outputs),
                                                        0,1,dtype=tf.float32)                   
        
        with tf.variable_scope('decoder'):
            recons_lstm_cells = _LSTMCells([self.n_hidden,self.input_dim],[lrelu,lrelu])
            self.recons_X,_ = tf.nn.dynamic_rnn(recons_lstm_cells, sample_Z, dtype=tf.float32)
 
        with tf.variable_scope('loss'):
            reduce_dims = np.arange(1,tf.keras.backend.ndim(self.X))
            recons_loss = tf.losses.mean_squared_error(self.X, self.recons_X)
            kl_loss = - 0.5 * tf.reduce_mean(1 + sigma_outputs - tf.square(mu_outputs) - tf.exp(sigma_outputs))
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
                    self.X: self.data_source.train
                    })
                print('round {}: with loss: {}'.format(i,mse_loss))
        self._arange_score(self.data_source.train)   
        
    
    def _arange_score(self,input_data):       
        input_all_losses = self.sess.run(self.all_losses,feed_dict={
                self.X: input_data                
                })
        self.anomaly_score = np.percentile(input_all_losses,(1-self.outlier_fraction)*100)
       
    def judge(self,test):
        all_test_loss = self.sess.run(self.all_losses,feed_dict={
                                    self.X: test                
                                    })
        result = map(lambda x: 1 if x< self.anomaly_score else -1,all_test_loss)

        return list(result)


    def plot_confusion_matrix(self):
        predict_label = self.judge(self.data_source.test)
        self.data_source.plot_confusion_matrix(self.data_source.test_label,predict_label,['Abnormal','Normal'],'LSTM_VAE Confusion-Matrix')

    
def main():

    lstm_vae = LSTM_VAE('dataset/data0.csv',['v0','v1'],z_dim=8,time_steps=16,outlier_fraction=0.01)    
    lstm_vae.train()  
    lstm_vae.plot_confusion_matrix() 

if __name__ == '__main__':
    main()

