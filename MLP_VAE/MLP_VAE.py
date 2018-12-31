# -*- coding: utf-8 -*-
"""
Schindler Liang

MLP Variational AutoEncoder for Anomaly Detection
reference: https://pdfs.semanticscholar.org/0611/46b1d7938d7a8dae70e3531a00fceb3c78e8.pdf
"""
import random
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.metrics import confusion_matrix


def lrelu(x, leak=0.2, name='lrelu'):
	return tf.maximum(x, leak*x)


def build_dense(input_vector,unit_no,activation):    
    return tf.layers.dense(input_vector,unit_no,activation=activation,
            kernel_initializer=tf.contrib.layers.xavier_initializer(),
            bias_initializer=tf.zeros_initializer())

class MLP_VAE:
    def __init__(self,input_dim,lat_dim, outliers_fraction):
       # input_paras:
           # input_dim: input dimension for X
           # lat_dim: latent dimension for Z
           # outliers_fraction: pre-estimated fraction of outliers in trainning dataset
        
        self.outliers_fraction = outliers_fraction # for computing the threshold of anomaly score       
        self.input_dim = input_dim
        self.lat_dim = lat_dim # the lat_dim can exceed input_dim    
        
        self.input_X = tf.placeholder(tf.float32,shape=[None,self.input_dim],name='source_x')
        
        self.learning_rate = 0.0005
        self.batch_size =  32
        # batch_size should be smaller than normal setting for getting
        # a relatively lower anomaly-score-threshold
        self.train_iter = 3000
        self.hidden_units = 128
        self._build_VAE()
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.pointer = 0
        
    def _encoder(self):
        with tf.variable_scope('encoder',reuse=tf.AUTO_REUSE):
            l1 = build_dense(self.input_X,self.hidden_units,activation=lrelu)
#            l1 = tf.nn.dropout(l1,0.8)
            l2 = build_dense(l1,self.hidden_units,activation=lrelu)
#            l2 = tf.nn.dropout(l2,0.8)          
            mu = tf.layers.dense(l2,self.lat_dim)
            sigma = tf.layers.dense(l2,self.lat_dim,activation=tf.nn.softplus)
            sole_z = mu + sigma *  tf.random_normal(tf.shape(mu),0,1,dtype=tf.float32)
        return mu,sigma,sole_z
        
    def _decoder(self,z):
        with tf.variable_scope('decoder',reuse=tf.AUTO_REUSE):
            l1 = build_dense(z,self.hidden_units,activation=lrelu)
#            l1 = tf.nn.dropout(l1,0.8)
            l2 = build_dense(l1,self.hidden_units,activation=lrelu)
#            l2 = tf.nn.dropout(l2,0.8)
            recons_X = tf.layers.dense(l2,self.input_dim)
        return recons_X


    def _build_VAE(self):
        self.mu_z,self.sigma_z,sole_z = self._encoder()
        self.recons_X = self._decoder(sole_z)
        
        with tf.variable_scope('loss'):
            KL_divergence = 0.5 * tf.reduce_sum(tf.square(self.mu_z) + tf.square(self.sigma_z) - tf.log(1e-8 + tf.square(self.sigma_z)) - 1, 1)
            mse_loss = tf.reduce_sum(tf.square(self.input_X-self.recons_X), 1)          
            self.all_loss =  mse_loss  
            self.loss = tf.reduce_mean(mse_loss + KL_divergence)
            
        with tf.variable_scope('train'):            
            self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
            

    def _fecth_data(self,input_data):        
        if (self.pointer+1) * self.batch_size  >= input_data.shape[0]:
            return_data = input_data[self.pointer*self.batch_size:,:]
            self.pointer = 0
        else:
            return_data =  input_data[ self.pointer*self.batch_size:(self.pointer+1)*self.batch_size,:]
            self.pointer = self.pointer + 1
        return return_data
    
     

    def train(self,train_X):
        for index in range(self.train_iter):
            this_X = self._fecth_data(train_X)
            self.sess.run([self.train_op],feed_dict={
                        self.input_X: this_X
                        })
        self.arrage_recons_loss(train_X)

        
    def arrage_recons_loss(self,input_data):
        all_losses =  self.sess.run(self.all_loss,feed_dict={
                self.input_X: input_data                  
                })
        self.judge_loss = np.percentile(all_losses,(1-self.outliers_fraction)*100)
                

    def judge(self,input_data):
        return_label = []
        for index in range(input_data.shape[0]):
            single_X = input_data[index].reshape(1,-1)
            this_loss = self.sess.run(self.loss,feed_dict={
                    self.input_X: single_X                  
                    })
            
            if this_loss < self.judge_loss:
                return_label.append(1)
            else:
                return_label.append(-1)
        return return_label
       
def plot_confusion_matrix(y_true, y_pred, labels,title):
    cmap = plt.cm.binary
    cm = confusion_matrix(y_true, y_pred)
    tick_marks = np.array(range(len(labels))) + 0.5
    np.set_printoptions(precision=2)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(4, 2), dpi=120)
    ind_array = np.arange(len(labels))
    x, y = np.meshgrid(ind_array, ind_array)
    intFlag = 0 
    for x_val, y_val in zip(x.flatten(), y.flatten()):
        #

        if (intFlag):
            c = cm[y_val][x_val]
            plt.text(x_val, y_val, "%d" % (c,), color='red', fontsize=8, va='center', ha='center')

        else:
            c = cm_normalized[y_val][x_val]
            if (c > 0.01):
                plt.text(x_val, y_val, "%0.2f" % (c,), color='red', fontsize=7, va='center', ha='center')
            else:
                plt.text(x_val, y_val, "%d" % (0,), color='red', fontsize=7, va='center', ha='center')
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
 
def mlp_vae_predict(train,test,test_label):
    mlp_vae = MLP_VAE(8,20,0.07)
    mlp_vae.train(train)
    mlp_vae_predict_label = mlp_vae.judge(test) 
    plot_confusion_matrix(test_label, mlp_vae_predict_label, ['anomaly','normal'],'MLP_VAE Confusion-Matrix')

def iforest_predict(train,test,test_label):
    from sklearn.ensemble import IsolationForest
    iforest = IsolationForest(max_samples = 'auto',
                                 behaviour="new",contamination=0.01)

    iforest.fit(train)
    iforest_predict_label = iforest.predict(test)
    plot_confusion_matrix(test_label, iforest_predict_label, ['anomaly','normal'],'iforest Confusion-Matrix')

def lof_predict(train,test,test_label):
    from sklearn.neighbors import LocalOutlierFactor
    lof = LocalOutlierFactor(novelty=True,contamination=0.01)
    lof.fit(train)
    lof_predict_label = lof.predict(test)
    plot_confusion_matrix(test_label, lof_predict_label, ['anomaly','normal'],'LOF Confusion-Matrix')

if __name__ == '__main__':
    train = np.load('data/train.npy') 
    test = np.load('data/test.npy')
    test_label = np.load('data/test_label.npy')
    mlp_vae_predict(train,test,test_label)
    iforest_predict(train,test,test_label)
    lof_predict(train,test,test_label)






