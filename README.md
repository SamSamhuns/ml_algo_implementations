# Machine Learning Algorithm Implementations

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Implementations of some popular machine learning algorithms

## Setup

Inside a virtual or conda env.

```shell
pip install -r requirements.txt
jupyter contrib nbextension install --user  # install non-python deps of jupyter nb extensions
jt -t chesterish -T -f roboto -fs 12 -cellw 95%  # set jupyter notebook themes
# enable nb extensions from the directory list menu after staring jupyter server
```

## Neural Networks

-   [Neural Network Implementation](ml_neural_networks)

-   [Deep Convolutional Neural Network for Image Classification](ml_deep_cnn_image_classification/README.md)

-   [Fully Connected Neural Network for handwritten digit recognition](ml_fcn_network_handwritten_digit_recognition/README.md)

-   [Recurrent Neural Networks](ml_rnn/README.md)

## Dimensionality Reduction

-   [PCA](ml_pca/README.md)

## Clustering

-   [Kmeans and Hierarchical Clustering](ml_kmeans_and_hierarchical_clustering/README.md)

## Classification and Regression

-   [Decision Trees](ml_decision_trees/README.md)

-   [K Nearest Neighbors](ml_knn/README.md)

-   [Linear and Logistic Regression](ml_linear_logistic_regression/README.md)

-   [Support Vector Machines](ml_svm/README.md)
