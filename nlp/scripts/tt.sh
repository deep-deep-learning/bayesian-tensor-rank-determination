#!/bin/bash

#may need smaller batch size to fit on 8gb gpu
BATCH_SIZE=128


for tensor_type in  'TensorTrain';
do for kl_mult in 0.0 1e-1 1e-2 1e-3 1e-4;
do python train.py  --n_epochs=100 --embedding ${tensor_type} --rank 20 --lr 0.0005 --kl-multiplier $kl_mult --rank-loss True --batch-size $BATCH_SIZE | tee logs/${tensor_type}_low_batch_${kl_mult}.txt;
done
done


python train.py  --n_epochs=100 --embedding full --lr 0.001 --batch-size $BATCH_SIZE | tee logs/full.txt;


#python train.py --gpu=0 --n_epochs=100 --embedding TensorTrain --rank 20 --lr 0.0005 --kl-multiplier 0.01 --rank-loss True 
#python train.py --gpu=0 --n_epochs=100 --embedding TensorTrainMatrix --rank 20 --lr 0.0005 --kl-multiplier 0.005 --rank-loss True --batch-size 220 
#python train.py --gpu=0 --n_epochs=100 --embedding Tucker --rank 5 --lr 0.0005 --kl-multiplier 0.005 --rank-loss True --batch-size 220 




#python train.py --gpu=0 --n_epochs=100 --embedding CP --rank 100 --lr 0.0005 | tee logs/cp.txt
#python train.py --gpu=0 --n_epochs=100 --embedding TensorTrain --rank 20 --lr 0.0005 | tee logs/tt.txt
#python train.py --gpu=0 --n_epochs=100 --embedding Tucker --rank 5 --lr 0.0005  --batch-size 220 | tee logs/tucker.txt
#python train.py --gpu=0 --n_epochs=100 --embedding TensorTrainMatrix --rank 20 --lr 0.0005  --batch-size 220 | tee logs/ttm.txt
