#!/bin/bash
./bench/tensorized_dlrm.sh --save-model='saved_models/CP' --tensor-type='CP' --use-gpu=0 --test-freq=248 --print-freq=124
./bench/tensorized_dlrm.sh --tensor-type='TensorTrain' --save-model='saved_models/TensorTrain' --use-gpu=0 --test-freq=248 --print-freq=124
./bench/tensorized_dlrm.sh --tensor-type='TensorTrainMatrix' --save-model='saved_models/TensorTrainMatrix' --use-gpu=0 --test-freq=248 --print-freq=124
./bench/tensorized_dlrm.sh --tensor-type='Tucker' --save-model='saved_models/Tucker' --use-gpu=0 --test-freq=248 --print-freq=124
