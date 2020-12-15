#!/bin/bash
export kl_mult=0.0
export no_kl_steps=50000
export lr=0.001
export CUDA_VISIBLE_DEVICES=""

source ~/.bashrc

for tensor_type in "TensorTrainMatrix" "CP" "Tucker"
do

export name="${tensor_type}_train_then_compress"
python train_then_compress.py  --memory-map  --arch-sparse-feature-size=128 --arch-mlp-bot="13-512-256-256-128" --arch-mlp-top="512-256-1" --data-generation=dataset --data-set=kaggle --raw-data-file=./input/train.txt --processed-data-file=./input/kaggleAdDisplayChallenge_processed.npz --loss-function='bce' --round-targets=True --test-num-workers=16 --memory-map --mini-batch-size=512 --nepochs=1 --optimizer="SGD" --learning-rate=${lr} --tensor-type="${tensor_type}" --test-freq=1024 --print-freq=512 --kl-multiplier=${kl_mult} --no-kl-steps=${no_kl_steps} > logs/${name}.log

done

