This README assumes you have already loaded the [Criteo Kaggle dataset](https://labs.criteo.com/2014/02/kaggle-display-advertising-challenge-dataset/) as described in the [Facebook DLRM repository](https://github.com/facebookresearch/dlrm). Most infrastructure and scripts in this repository are taken from the original DLRM repository.

The dataset should be located in the folder `input`.

## Setup

To install the tensorized embedding package from the root directory using pip:
```
cd torch_bayesian_tensor_layers
pip install -e .
```
This should set up the tensorized layers package. The tensorized embedding package does not compute the full embedding, but instead only selects the necessary tensor slices. This lookup can be further optimized by avoiding redundant computations.




## Issues?

Feel free to contact `colepshawkins@gmail.com` with any questions, or raise an issue.
