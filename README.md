

## Setup

The DLRM example has a separate requirements file, with installation instructions in the `dlrm` directory. The MNIST and NLP examples have fewer requirements, which can be installed using
```
conda env update -f requirements.yml

```
This will create the `tensor_layers` conda environment.

To install the tensorized embedding package from the root directory using pip:
```
cd torch_bayesian_tensor_layers
pip install -e .
```
This should set up the tensorized layers package. The tensorized embedding package does not compute the full embedding, but instead only selects the necessary tensor slices. This lookup can be further optimized by avoiding redundant computations. The tensorized forward propagation for the MLP often reforms the full tensor, which is suboptimal.



## Issues?

Feel free to contact `colepshawkins@gmail.com` with any questions, or raise an issue.
