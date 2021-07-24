#!/bin/bash

#may need smaller batch size to fit on 8gb gpu
python train.py --gpu=0 --n_epochs=100 --embedding CP --rank 100 --lr 0.0005 --kl-multiplier 0.005 --rank-loss True > logs/cp_lu.txt
python train.py --gpu=0 --n_epochs=100 --embedding TensorTrain --rank 20 --lr 0.0005 --kl-multiplier 0.01 --rank-loss True > logs/tt_lu.txt
python train.py --gpu=0 --n_epochs=100 --embedding TensorTrainMatrix --rank 20 --lr 0.0005 --kl-multiplier 0.005 --rank-loss True --batch-size 220 > logs/ttm.txt
python train.py --gpu=0 --n_epochs=100 --embedding Tucker --rank 5 --lr 0.0005 --kl-multiplier 0.005 --rank-loss True --batch-size 220 > logs/tucker.txt




python train.py --gpu=0 --n_epochs=100 --embedding CP --rank 100 --lr 0.0005 > logs/cp.txt
python train.py --gpu=0 --n_epochs=100 --embedding TensorTrain --rank 20 --lr 0.0005  > logs/tt.txt
python train.py --gpu=0 --n_epochs=100 --embedding Tucker --rank 5 --lr 0.0005  --batch-size 220 > logs/tucker.txt
python train.py --gpu=0 --n_epochs=100 --embedding TensorTrainMatrix --rank 20 --lr 0.0005  --batch-size 220 > logs/ttm.txt