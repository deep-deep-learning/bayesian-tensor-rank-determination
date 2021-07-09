This README assumes you have already loaded the [Criteo Kaggle dataset](https://labs.criteo.com/2014/02/kaggle-display-advertising-challenge-dataset/) as described in the [Facebook DLRM repository](https://github.com/facebookresearch/dlrm) which forms most of the infrastructure for this repository.

The dataset should be located in the folder `input`.

## Setup

Install the requirements using 
```
conda env update -f conda_reqs.yml
```
Next install the tensorized embedding package by moving from the root directory to the package directory and using pip:
```
cd torch_bayesian_tensor_layers
pip install -e .
```
This should set up the tensorized embedding package. The tensorized embedding package does not compute the full embedding, but instead only selects the necessary tensor slices. This lookup can be further optimized by avoiding redundant computations.

## Running the tensorized models with rank determination

The appropriate scripts are all located in the directory `bench`. Assuming you have set up a conda environment using the pip install above, you should be able to run the scripts in `bench` from the root directory.

Most script names are self-explanatory. The tensorized model scripts will print accuracy and ranks at each validation/test checkpoint. 

For example `run_cp_hc.sh` runs the half-cauchy CP model hyperparameter search.

In order to run the train-then-compress approach via `compress_then_train.sh` you will need to save a DLRM model.


## Issues?

Feel free to contact `colepshawkins@gmail.com` with any questions, or raise an issue.
