# Structured Relevance Determination and Channel-Additive Gaussian Processes for Image Data

Implementation of kernels that incorporate a form of hyperparameter-sharing tailored to image-valued inputs and code for experiments.

The accompanying paper is available [here](https://arxiv.org/abs/2610.25385).

More specifically, it includes:
- hyperparameter-sharing accross groups of adjacent pixels (Automatic Patch Relevance Detection, APRD),
- hyperparameter-sharing across colour channels (Automatic Channel Relevance Detection, ACRD),
- channel-additive Gaussian processes that utilise hyperparameter-sharing,
- integration of ACRD into convolutional Gaussian process models,
- combining such convolutional Gaussian process models with a model that is based on channel-additivity and hyperparameter-sharing.

## Experiments

Experiments on MNIST (with 10 percent of training data and full training data) and on a [planes in satellity imagery dataset](https://www.kaggle.com/datasets/rhammell/planesnet) can be run from the respective directories.
The necessary steps for setting up the datasets, as well as the specific commands to run the experiments from the paper are provided there.

All experiments were run on an NVIDIA GeForce RTX 3060.
