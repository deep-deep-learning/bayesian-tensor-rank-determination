#!/bin/bash

#may need smaller batch size to fit on 8gb gpu


for tensor_type in  'TensorTrainMatrix';
do for kl_mult in 0.0 1.0 0.5;
do python train.py   --model-type ${tensor_type} --rank 20  --kl-multiplier $kl_mult --rank-loss True | tee logs/${tensor_type}_${kl_mult}.txt;
done
done



