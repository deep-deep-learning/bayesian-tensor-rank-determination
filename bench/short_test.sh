#!/bin/bash
./bench/tensorized_dlrm.sh --save-model='saved_models/CP' --tensor-type='CP' --use-gpu=0 --test-freq=248 --print-freq=124 --optimizer='Adagrad' --learning-rate=1.0 