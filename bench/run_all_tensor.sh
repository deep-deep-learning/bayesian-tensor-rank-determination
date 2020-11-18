#!/bin/bash
for tensor_type in "CP" "TensorTrain" "TensorTrainMatrix" "Tucker"
do for no_kl_steps in 50000 100000 150000;
do for lr in 1.0 0.1 0.01 0.001;
do
	name="${tensor_type}_warmup_${no_kl_steps}_adagrad_${lr}"
	./bench/tensorized_dlrm.sh --optimizer='Adagrad' --learning-rate=0.1 --save-model="saved_models/${name}" --tensor-type="${tensor_type}" --use-gpu=1 --test-freq=10240 --print-freq=1024 --kl-multiplier=1.0 --no-kl-steps=${no_kl_steps} > logs/${name}.log
done
done
done