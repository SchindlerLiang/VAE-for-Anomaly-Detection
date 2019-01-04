
LSTM_VAE used for Univariate Time Series Anomaly Detection;
[reference1](https://www.researchgate.net/publication/304758073_LSTM-based_Encoder-Decoder_for_Multi-sensor_Anomaly_Detection);
[reference2](https://github.com/twairball/keras_lstm_vae);
[reference3](https://arxiv.org/pdf/1711.00614.pdf);

The dataset used is the [MTSAD](https://github.com/jsonbruce/MTSAnomalyDetection). We use StandardScaler and MinMaxScaler to preprocess the initial data. Then we re-set the dataset to be 3_dimensional with time_steps of 6. For each sample, if one in the 6_timesteps is labeled as abnormal, then the corresponding 3_dimensional sample is labeled as ABNORMAL;

In total, there are 1326 abnormal samples and 7419 normal samples. We randomly select 6000 normal samples as train dataset, 1419 normal samples and 1326 abnormal samples as test dataset. As a result, the abnormal samples constitute 48.3% of the test dataset.

The LSTM_VAE algorithm is surprisingly difficult to train, compared with both standard LSTM and ordinary VAE. One possible explanation could be that the kl_divergence and the mse_loss (between input and the reconstructed input) are kind of in contradiction with each other.

The confusion_matrix of the test dataset are presented as:

![Confusion_Matrix for LSTM_VAE](https://github.com/SchindlerLiang/VAE-for-Anomaly-Detection/blob/master/LSTM_VAE_univariate/LSTM_VAE.png)

It can be concluded from above that LSTM_VAE is capable of capturing the outliers (anomaly) in the test dataset.


