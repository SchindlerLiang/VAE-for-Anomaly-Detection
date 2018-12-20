# -*- coding: utf-8 -*-
"""
Schindler Liang

MLP Variational AutoEncoder for Anomaly Detection
reference: https://pdfs.semanticscholar.org/0611/46b1d7938d7a8dae70e3531a00fceb3c78e8.pdf
"""

import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy

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
        
        self.outliers_fraction = outliers_fraction        
        self.lat_dim = lat_dim        
        self.input_dim = input_dim
        
        self.sample_length = 100 #  drawn sample_length samples from p(x/z) for computing average reconstruction error

        self.input_X = tf.placeholder(tf.float32,shape=[None,self.input_dim],name='source_x')
        
        self.epsilon = 1e-8  # for computing np.log(pdf+self.epsilon) in case of 0
        
        self.keep_prob = 0.9
        self.learn_rate = 0.00001
        self.batch_size = 256 
        # batch_size should be smaller than normal setting for getting
        # a relatively lower anomaly-score-threshold
        self.train_iter = 1600        
        self.hidden_units = 128
        
        self._build_VAE()
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.pointer = 0
        
        
    def _encoder(self):
        with tf.variable_scope('encoder',reuse=tf.AUTO_REUSE):
            l1 = build_dense(self.input_X,self.hidden_units,tf.nn.elu)
            l1 = tf.nn.dropout(l1,keep_prob=self.keep_prob)
            l2 = build_dense(l1,self.hidden_units,tf.nn.elu)
            l2= tf.nn.dropout(l2,keep_prob=self.keep_prob)
            mu = tf.layers.dense(l2,self.lat_dim)
            sigma = build_dense(l2,self.lat_dim,tf.nn.softplus)
            z = mu + sigma * tf.random_normal(tf.shape(mu),0,1,dtype=tf.float32)
        return mu,sigma,z
        
    def _decoder(self,z):
        with tf.variable_scope('decoder',reuse=tf.AUTO_REUSE):
            l1 = build_dense(z,self.hidden_units,tf.nn.elu)
            l1 = tf.nn.dropout(l1,keep_prob=self.keep_prob)            
            l2 = build_dense(l1,self.hidden_units,tf.nn.elu)
            l2 = tf.nn.dropout(l2,keep_prob=self.keep_prob)                      
            reconx_x_mu = tf.layers.dense(l2,self.input_dim)
            reconx_x_sigma = tf.layers.dense(l2,self.input_dim,activation=tf.nn.softplus)
            recons_X = reconx_x_mu + reconx_x_sigma * tf.random_normal(tf.shape(reconx_x_mu),0,1,dtype=tf.float32)            
        return recons_X


    def _build_VAE(self):
        mu_z,sigma_z,sole_z = self._encoder()
        self.recons_X = self._decoder(sole_z)
        
        with tf.variable_scope('loss'):
            normal_01_distribution = tf.distributions.Normal(loc=tf.zeros_like(mu_z),scale=tf.ones_like(mu_z))
            z_distribution = tf.distributions.Normal(loc=mu_z,scale=sigma_z)
            self.all_KL_divergences = tf.distributions.kl_divergence(z_distribution,normal_01_distribution)
            KL_divergence = tf.reduce_mean(self.all_KL_divergences)
            mse_loss = tf.losses.mean_squared_error(self.input_X,self.recons_X)            
            self.loss = mse_loss + KL_divergence
        with tf.variable_scope('train'):
            self.train_op = tf.train.AdamOptimizer(self.learn_rate).minimize(self.loss)
            
        
        with tf.variable_scope('recons_error_loss'): #sample z from normal(z_mu,z_sigma)
            sample_Zs = z_distribution.sample(self.sample_length)
            sample_Zs = tf.transpose(sample_Zs,[1,0,2]) 
            sample_Zs = tf.reshape(sample_Zs,[-1,self.lat_dim])
            test_recons_X = self._decoder(sample_Zs) # get sample_length recons_X from decoder
            test_recons_X = tf.reshape(test_recons_X,[-1,self.sample_length,self.input_dim])
            new_X = tf.expand_dims(self.input_X,1)
            new_X = tf.tile(new_X,[1,self.sample_length,1])
            self.recons_error = tf.reduce_mean(  tf.square(new_X - test_recons_X),1) 
            self.error_mu,self.error_sigma = tf.nn.moments(self.recons_error,0)
                     
    def _fecth_data(self,train_X):        
        if train_X.shape[0] < self.batch_size:
            return train_X
        else:
            if (self.pointer+1) * self.batch_size  >= train_X.shape[0]:
                
                return_data = train_X[self.pointer*self.batch_size:,:]
                self.pointer = 0
            else:
                return_data =  train_X[ self.pointer*self.batch_size:(self.pointer+1)*self.batch_size,:]
                self.pointer = self.pointer + 1
            return return_data
        
    def _train(self,train_X):
        for _ in range(self.train_iter):
            this_X = self._fecth_data(train_X)
            self.sess.run([self.train_op],feed_dict={
                    self.input_X:this_X
                    })
        
        # anomaly score = average KL-divergence - average log(pdf(recon_X))
        kl_divergence,all_errors,train_error_mean,train_error_std = self.sess.run([
                self.all_KL_divergences,
                self.recons_error,self.error_mu,
                self.error_sigma],feed_dict={
                self.input_X: train_X                
                })
        self.error_gaussion = scipy.stats.norm(loc=train_error_mean,scale=train_error_std)
        prob_prod = np.prod(self.error_gaussion.pdf(all_errors) + self.epsilon, axis=1)
        log_prob_mean = np.log(prob_prod)        
        kl_divergence = np.mean(kl_divergence,axis=1)
        anomaly_scores = kl_divergence.ravel() - log_prob_mean.ravel()
        self.judge_score = np.percentile(anomaly_scores,(1-self.outliers_fraction)*100)
    
    def judge(self,X):
        self.this_kl_divergence, self.this_error = self.sess.run(
                    [self.all_KL_divergences,
                     self.recons_error],feed_dict={
                    self.input_X: X                    
                    })
        self.this_log_prob_array = np.prod(self.error_gaussion.pdf(self.this_error)+ self.epsilon)
        
        self.this_score = np.mean(self.this_kl_divergence) - np.log(self.this_log_prob_array)
        
        if self.this_score >  self.judge_score:
            return 2
        else:
            return 1

        
def main():       
    train = np.load('data/train.npy')
    test = np.load('data/test.npy')
    test_label = np.load('data/test_label.npy').ravel()
    
    mlp_vae = MLP_VAE(3,1,0.02)
    mlp_vae._train(train)
    
    result = np.array([])
    for i in range(test.shape[0]):
        this_X = test[i][np.newaxis,:]
        this_judge = mlp_vae.judge(this_X)
        result = np.append(result,this_judge)
    
    label = 2
    plt.plot(test_label[test_label==label],label='source')
    plt.plot(result[test_label==label],label='predict')
    plt.legend()



if __name__ == '__main__':
    main()





    
