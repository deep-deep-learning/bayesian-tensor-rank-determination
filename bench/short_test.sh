#!/bin/bash
./bench/tensorized_dlrm.sh --save-model='saved_models/CP' --tensor-type='CP' --optimizer='Adam' --use-gpu=0 --test-freq=248 --print-freq=124 --learning-rate=0.005