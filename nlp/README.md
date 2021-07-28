The experimental setup is taken from existing the [tt-pytorch](https://github.com/KhrulkovV/tt-pytorch) repo.


## Setup 

Install the requirements and set up the conda environment:
```
conda env update -f requirements.yml
conda activate tensor_layers
```

Install spacy `en` for the tokenizer:
```
python -m spacy download en
```
This setup has been tested on a clean g4dn-xlarge AWS instance running Ubuntu 18.04 (deep learning AMI).

## Run the code

The appropriate scripts are all in the scripts directory. They call the `train.py` file with the appropriate command line arguments.

For example, to run the CP format:
```
bash scripts/cp.sh
```
The TT,TTM, and Tucker formats are similar.
