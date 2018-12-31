
MLP_VAE used for anomaly detection;
[reference](https://pdfs.semanticscholar.org/0611/46b1d7938d7a8dae70e3531a00fceb3c78e8.pdf);

The dataset used is the [HTRU2 Data Set](http://archive.ics.uci.edu/ml/datasets/HTRU2). This is an unbanlaced dataset, where samples with Class 1 constitutes less than 10% of the entire dataset, which is treated as anomaly class;

All the dimensions are preprocessed by sklearn StandardScaler and MinMaxScaler to better fit for MLP_VAE;

The test results of MLP_VAE,IForest and LOF are presented as follows:

![Confusion_Matrix for MLP_VAE](https://github.com/SchindlerLiang/VAE-for-Anomaly-Detection/blob/master/MLP_VAE/img/MLP_VAE.png)

![Confusion_Matrix for Iforest](https://github.com/SchindlerLiang/VAE-for-Anomaly-Detection/blob/master/MLP_VAE/img/iforest.png)

![Confusion_Matrix for LOF](https://github.com/SchindlerLiang/VAE-for-Anomaly-Detection/blob/master/MLP_VAE/img/lof.png)

The outliers_fraction for MLP_VAE are specially set to be different for better computing the anomaly score. It can be seen from above that MLP_VAE can obtain even results with IForest and LOF;
