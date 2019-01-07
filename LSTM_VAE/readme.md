## Reference
LSTM_VAE used for Multivariate Time Series Anomaly Detection;
[reference1](https://www.researchgate.net/publication/304758073_LSTM-based_Encoder-Decoder_for_Multi-sensor_Anomaly_Detection);
[reference2](https://github.com/twairball/keras_lstm_vae);
[reference3](https://arxiv.org/pdf/1711.00614.pdf);

## Prerequisites
* Python 3.3+
* Tensorflow 1.12.0
* Sklearn 0.20.1
* Numpy 1.15.4
* Pandas 0.23.4
* Matplotlib 3.0.2

## Dataset and Preprocessing
The dataset used is the [MTSAD](https://github.com/jsonbruce/MTSAnomalyDetection), which has 2 dimensions.
We use StandardScaler and MinMaxScaler to preprocess the initial data. Then we re-set the dataset to be 3_dimensional with time_steps of 10. 
For each sample, if ANY ONE in the 10_timesteps is labeled as abnormal, then the corresponding 3_dimensional sample is labeled as ABNORMAL;

In total, there are 55 abnormal samples and 8661 normal samples. We randomly select 8000 normal samples as train dataset, 661 normal samples and 55 abnormal samples as test dataset. As a result, the abnormal samples constitute only 7.7% of the test dataset.

`LSTM_VAE should be trained on NORMAL Dataset. However, dataset with only a few ABNORMAL samples is also acceptable, since we can adjust the hyper-parameter outliers_fraction, which may slightly influnce the detection score.`

## Result
The confusion_matrix of the test dataset are presented as:

![Confusion_Matrix for LSTM_VAE](https://github.com/SchindlerLiang/VAE-for-Anomaly-Detection/blob/master/LSTM_VAE/LSTM_VAE.png)

It can be concluded from above that LSTM_VAE is capable of capturing most of the outliers (anomaly) in the test dataset.

