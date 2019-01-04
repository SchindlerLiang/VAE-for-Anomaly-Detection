# -*- coding: utf-8 -*-
"""
One simple Implementation of LSTM_VAE based algorithm for Anomaly Detection in for Time Series;

Author: Schindler Liang

Reference:
    https://www.researchgate.net/publication/304758073_LSTM-based_Encoder-Decoder_for_Multi-sensor_Anomaly_Detection
    https://github.com/twairball/keras_lstm_vae
    https://arxiv.org/pdf/1711.00614.pdf
    
"""
import pandas as pd
import numpy as np
import tensorflow as tf
#from tensorflow.contrib.rnn import LayerNormBasicLSTMCell, DropoutWrapper
from tensorflow.nn.rnn_cell import MultiRNNCell, LSTMCell
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
def lrelu(x, leak=0.2, name='lrelu'):
	return tf.maximum(x, leak*x)

def lstm_unit(unit_number,is_train=False): 
    lstm_cell = LSTMCell(unit_number, name='basic_lstm_cell',initializer=tf.contrib.layers.xavier_initializer(),state_is_tuple=True)               
    return lstm_cell

class LSTM_VAE(object):
    def __init__(self,input_dim,z_dim,time_steps,outlier_fraction):
        self.outlier_fraction = outlier_fraction
        self.n_hidden = 32
        self.batch_size = 32
        self.learning_rate = 0.0005
        self.train_iters = 60000
        
        self.input_dim = input_dim
        self.z_dim = z_dim
        self.time_steps = time_steps
    
        self.pointer = 0 
        self.anomaly_score = 0
        self.sess = tf.Session()
        self._build_network()
        self.sess.run(tf.global_variables_initializer())
        
    def _build_network(self):
        with tf.variable_scope('ph',reuse=tf.AUTO_REUSE):
            self.X = tf.placeholder(tf.float32,shape=[None,self.time_steps,self.input_dim],name='input_X')
        
        with tf.variable_scope('encoder'):
            en_lstm_cells = MultiRNNCell([lstm_unit(unit,self.is_train) for unit in [self.n_hidden,self.n_hidden]],state_is_tuple=True)
            outputs,_ = tf.nn.dynamic_rnn(en_lstm_cells, self.X, dtype=tf.float32)
            outputs = tf.reshape(outputs,[tf.shape(self.X)[0],self.time_steps*self.n_hidden])
            mu_z = tf.layers.dense(outputs,self.z_dim,activation=lrelu)
            sigma_z = tf.layers.dense(outputs,self.z_dim,activation=tf.nn.softplus)
            
        with tf.variable_scope('decoder'):
            sample_op = lambda x: mu_z + sigma_z * tf.random_normal(tf.shape(mu_z),0,1,dtype=tf.float32)
            sample_Z = tf.map_fn(sample_op,np.arange(self.time_steps).astype(np.float32))
            sample_Z = tf.transpose(sample_Z,[1,0,2])
            de_lstm_cells = MultiRNNCell([lstm_unit(unit,self.is_train) for unit in [self.n_hidden,self.input_dim]],state_is_tuple=True)
            self.recons_X,_ = tf.nn.dynamic_rnn(de_lstm_cells, sample_Z, dtype=tf.float32)

 
        with tf.variable_scope('loss'):
            recons_loss = tf.losses.mean_squared_error(self.X, self.recons_X)
            kl_loss = - 0.5 * tf.reduce_mean(1 + sigma_z - tf.square(mu_z) - tf.exp(sigma_z))
            self.opt_loss = recons_loss + kl_loss
            reduce_dims = np.arange(1,tf.keras.backend.ndim(self.X))
            self.all_losses = tf.reduce_sum(tf.square(self.X - self.recons_X),reduction_indices=reduce_dims)

        with tf.variable_scope('train'):
            self.uion_train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.opt_loss)
            
            
    def train(self,train_X):
        for i in range(self.train_iters):            
            this_X = self._fetch_data(train_X)
            self.sess.run([self.uion_train_op],feed_dict={
                    self.X: this_X
                    })
            if i % 50 ==0:
                mse_loss = self.sess.run([self.opt_loss],feed_dict={
                    self.X: train_X
                    })
                print('round {}: with loss: {}'.format(i,mse_loss))
        self._arange_score(train_X)   
        
    def _fetch_data(self,train_X):
        if train_X.shape[0] < self.batch_size:
            return_train = train_X
        else:
            if (self.pointer + 1) * self.batch_size >= train_X.shape[0]-1:
                self.pointer = 0
                return_train = train_X[self.pointer * self.batch_size:,]
            else:
                self.pointer = self.pointer + 1
                return_train = train_X[self.pointer * self.batch_size:(self.pointer + 1) * self.batch_size,]
        if return_train.ndim < train_X.ndim:
            return_train = np.expand_dims(return_train,0)
        return return_train
    
    def _arange_score(self,input_data):       
        input_all_losses = self.sess.run(self.all_losses,feed_dict={
                self.X: input_data                
                })
        self.anomaly_score = np.percentile(input_all_losses,(1-self.outlier_fraction)*100)
       
    def judge(self,test):
        result = np.array([])
        for index in range(test.shape[0]):
            this_X = test[index][np.newaxis,:,:]
            this_loss = self.sess.run(self.all_losses,feed_dict={
                    self.X: this_X                
                    })
            if float(this_loss) >= self.anomaly_score:
                result = np.append(result,-1)
            else:
                result = np.append(result,1)
        return result

def plot_confusion_matrix(y_true, y_pred, labels,title):
    cmap = plt.cm.binary
    cm = confusion_matrix(y_true, y_pred)
    tick_marks = np.array(range(len(labels))) + 0.5
    np.set_printoptions(precision=2)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(8, 4), dpi=120)
    ind_array = np.arange(len(labels))
    x, y = np.meshgrid(ind_array, ind_array)
    intFlag = 0 
    for x_val, y_val in zip(x.flatten(), y.flatten()):
        #

        if (intFlag):
            c = cm[y_val][x_val]
            plt.text(x_val, y_val, "%d" % (c,), color='red', fontsize=10, va='center', ha='center')

        else:
            c = cm_normalized[y_val][x_val]
            if (c > 0.01):
                plt.text(x_val, y_val, "%0.2f" % (c,), color='red', fontsize=10, va='center', ha='center')
            else:
                plt.text(x_val, y_val, "%d" % (0,), color='red', fontsize=10, va='center', ha='center')
    if(intFlag):
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
    else:
        plt.imshow(cm_normalized, interpolation='nearest', cmap=cmap)
    plt.gca().set_xticks(tick_marks, minor=True)
    plt.gca().set_yticks(tick_marks, minor=True)
    plt.gca().xaxis.set_ticks_position('none')
    plt.gca().yaxis.set_ticks_position('none')
    plt.grid(True, which='minor', linestyle='-')
    plt.gcf().subplots_adjust(bottom=0.15)
    plt.title(title)
    plt.colorbar()
    xlocations = np.array(range(len(labels)))
    plt.xticks(xlocations, labels)
    plt.yticks(xlocations, labels)
    plt.ylabel('Index of True Classes')
    plt.xlabel('Index of Predict Classes')
    plt.show()
    
def main():
    train = np.load('dataset/lstm_train.npy')
    lstm_vae = LSTM_VAE(1,30,6,0.05)
    
    lstm_vae.train(train)
    test = np.load('dataset/lstm_test.npy')
    test_label = np.load('dataset/lstm_test_label.npy')
       
    predict_label = lstm_vae.judge(test)
    plot_confusion_matrix(test_label, predict_label, ['abnormal','normal'],'LSTM_VAE Confusion_Matrix')


if __name__ == '__main__':
    main()

