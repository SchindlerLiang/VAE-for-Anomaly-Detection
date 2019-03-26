## Reference
High-frequency Multivariate Time Series Anomaly Detection based on IndRNN with AutoEncoder(both AE and VAE);
[reference1](https://github.com/twairball/keras_lstm_vae). The IndRNN implementation is from 
[reference2](https://github.com/batzner/indrnn);


## Prerequisites
* Python 3.3+
* Tensorflow 1.12.0
* Sklearn 0.20.1
* Numpy 1.15.4
* Pandas 0.23.4
* Matplotlib 3.0.2

## Dataset and Preprocessing
The dataset used is the [MTSAD](https://github.com/jsonbruce/MTSAnomalyDetection), which has 2 dimensions. 
Then we re-set the dataset to be 3_dimensional with time_steps of 16. The detailed preprecessing process can be found at 
the LSTM_VAE chapter[reference3](https://github.com/SchindlerLiang/VAE-for-Anomaly-Detection/blob/master/LSTM_VAE/utils.py).

IndRNN_(V)AE algorithm should be trained on the Normal samples. In this algorithm, we present two score-functions for accessing the test_data. judge_anomaly() for anomaly detection and judge_health() for healthy accessment, which may be of use in high-frequency industry sensors.


## Network Structure
The Structure of the network presented here 

![Network Structure for IndRNN_(V)AE](https://github.com/SchindlerLiang/VAE-for-Anomaly-Detection/blob/master/INDRNN_(V)AE/graph.png)

Note that we use both AE and VAE structure, with the thoughts of keeping time-dependent information by AE and maitaining variability by VAE. 
