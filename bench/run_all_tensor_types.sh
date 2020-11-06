#!/bin/bash
./bench/tensorized_dlrm.sh --tensor-type='CP'
./bench/tensorized_dlrm.sh --tensor-type='TensorTrain'
./bench/tensorized_dlrm.sh --tensor-type='TensorTrainMatrix'
./bench/tensorized_dlrm.sh --tensor-type='Tucker'