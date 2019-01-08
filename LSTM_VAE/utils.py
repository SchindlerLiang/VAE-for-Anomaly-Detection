import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler,MinMaxScaler

'''
time_steps = 10
'''
class Data_Hanlder(object):
    
    def __init__(self):
        
        self.data = pd.read_csv('dataset/data0.csv',index_col=0)
        self.columns = ['v0','v1']
        self.data['Class'] = 0
        self.data['Class'] = self.data['result'].apply(lambda x: 1 if x=='normal' else -1)
        self.data[self.columns] = self.data[self.columns].shift(-1) - self.data[self.columns]
        self.data = self.data.dropna(how='any')
               
    def _process_source_data(self):
 
        self.data_scale()
        self.data_arrage()
        self.split_save_data()
        
    def data_scale(self):

        standscaler = StandardScaler()
        mscaler = MinMaxScaler(feature_range=(0,1))
        self.data[self.columns] = standscaler.fit_transform(self.data[self.columns])
        self.data[self.columns] = mscaler.fit_transform(self.data[self.columns])


    def data_arrage(self):

        d_array = self.data[self.columns].values
        self.all_data = np.array([[a,b,c,d,e,f,g,h,i,j] 
                for a,b,c,d,e,f,g,h,i,j in 
                zip(d_array[:-9],d_array[1:-8],d_array[2:-7],
                    d_array[3:-6],d_array[4:-5],d_array[5:-4],
                    d_array[6:-3],d_array[7:-2],d_array[8:-1],
                    d_array[9:]
                    )])
        self.labels = np.array([])
        for index in range(self.data.shape[0]-9):
            data_label = self.data['Class'].values[index:index+10]
            if np.any(data_label==-1):
                self.labels = np.append(self.labels,-1)
            else:
                self.labels = np.append(self.labels,1)
        
    def split_save_data(self):
        normal = self.all_data[self.labels==1]
        abnormal = self.all_data[self.labels==-1]
        train = normal[:8000,:]
        test = np.concatenate([normal[8000:,:],abnormal],axis=0)
        test_label = np.concatenate([np.ones(normal[8000:,:].shape[0]),-np.ones(abnormal.shape[0])])        
        np.save('dataset/lstm_train.npy',train)
        np.save('dataset/lstm_test.npy',test)
        np.save('dataset/lstm_test_label.npy',test_label)

    def get_data(self):
        train = np.load('dataset/lstm_train.npy')
        test = np.load('dataset/lstm_test.npy')
        test_label = np.load('dataset/lstm_test_label.npy')        
        return train,test,test_label

