import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler,MinMaxScaler
import os
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

'''
time_steps = 10
'''
class Data_Hanlder(object):
    
    def __init__(self,dataset_name,columns,time_steps):
        self.time_steps = time_steps        
        self.data = pd.read_csv(dataset_name,index_col=0)
        self.columns = columns
        
        self.data['Class'] = 0
        self.data['Class'] = self.data['result'].apply(lambda x: 1 if x=='normal' else -1)
        self.data[self.columns] = self.data[self.columns].shift(-1) - self.data[self.columns]
        self.data = self.data.dropna(how='any')
        self.pointer = 0
        self.train = np.array([])
        self.test = np.array([])
        self.test_label = np.array([])
        
        
        self.split_fraction = 0.2
        
        
    def _process_source_data(self):
 
        self._data_scale()
        self._data_arrage()
        self._split_save_data()
        
    def _data_scale(self):

        standscaler = StandardScaler()
        mscaler = MinMaxScaler(feature_range=(0,1))
        self.data[self.columns] = standscaler.fit_transform(self.data[self.columns])
        self.data[self.columns] = mscaler.fit_transform(self.data[self.columns])


    def _data_arrage(self):
        
        self.all_data = np.array([])
        self.labels = np.array([])
        d_array = self.data[self.columns].values  
        class_array = self.data['Class'].values
        for index in range(self.data.shape[0]-self.time_steps+1):
            this_array = d_array[index:index+self.time_steps].reshape((-1,self.time_steps,len(self.columns)))
            time_steps_label = class_array[index:index+self.time_steps]
            if np.any(time_steps_label==-1):
                this_label = -1
            else:
                this_label = 1
            if self.all_data.shape[0] == 0:
                self.all_data = this_array
                self.labels = this_label                    
            else:
                self.all_data = np.concatenate([self.all_data,this_array],axis=0)
                self.labels = np.append(self.labels,this_label)
        
    def _split_save_data(self):
        normal = self.all_data[self.labels==1]
        abnormal = self.all_data[self.labels==-1]
        
        split_no =   normal.shape[0] -  abnormal.shape[0]    
        
        self.train = normal[:split_no,:]
        self.test = np.concatenate([normal[split_no:,:],abnormal],axis=0)
        self.test_label = np.concatenate([np.ones(normal[split_no:,:].shape[0]),-np.ones(abnormal.shape[0])])        
        np.save('dataset/train.npy',self.train)
        np.save('dataset/test.npy',self.test)
        np.save('dataset/test_label.npy',self.test_label)

    def _get_data(self):
        if os.path.exists('dataset/train.npy'):
            self.train = np.load('dataset/train.npy')
            self.test = np.load('dataset/test.npy')
            self.test_label = np.load('dataset/test_label.npy')        
        if self.train.ndim ==3:
            if self.train.shape[1] == self.time_steps and self.train.shape[2] != len(self.columns):
                return 0
        self._process_source_data()


    def fetch_data(self,batch_size):
        if self.train.shape[0] == 0:
            self._get_data()
            
        if self.train.shape[0] < batch_size:
            return_train = self.train
        else:
            if (self.pointer + 1) * batch_size >= self.train.shape[0]-1:
                self.pointer = 0
                return_train = self.train[self.pointer * batch_size:,]
            else:
                self.pointer = self.pointer + 1
                return_train = self.train[self.pointer * batch_size:(self.pointer + 1) * batch_size,]
        if return_train.ndim < self.train.ndim:
            return_train = np.expand_dims(return_train,0)
        return return_train
    
    def plot_confusion_matrix(self,y_true, y_pred, labels,title):
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