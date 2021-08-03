#!/bin/bash

#may need smaller batch size to fit on 8gb gpu


for tensor_type in  'CP';
do for kl_mult in 1e-3 5e-4 1e-4 5e-5 1e-5;
do python train.py   --model-type ${tensor_type} --rank 50  --kl-multiplier $kl_mult --rank-loss True | tee logs/${tensor_type}_${kl_mult}.txt;
done
done



