This README assumes you have already loaded the dataset as detailed in the repository `https://github.com/facebookresearch/dlrm` which forms the backbone of this implementation and contains a much more thorough set of directions for running the DLRM model.

## Setup

Install the requirements using 
```
pip install -r requirements.txt
```
Next install the tensorized embedding package by moving from the root directory to the package directory and using pip:
```
cd torch_bayesian_tensor_layers
pip install -e .
```
This should set up the tensorized embedding package. The tensorized embedding package does not compute the full embedding, but instead only selects the necessary tensor slices. This lookup can be further optimized by avoiding redundant computations.

## Running the tensorized models with rank determination

The appropriate scripts are all located in the directory `bench`. Assuming you have set up a conda environment using the pip install above, you should be able to run the scripts in `bench` from the root directory.

Most script names are self-explanatory. 

For example `run_cp_hc.sh` runs the half-cauchy CP model hyperparameter search. 

In order to run the train-then-compress approach via `compress_then_train.sh` you will need to save a DLRM model.


## Issues?

Feel free to contact `colepshawkins@gmail.com` with any questions, or raise an issue.
