#!/bin/bash
./bench/tensorized_dlrm.sh --load-model='saved_models/CP' --tensor-type='CP' --use-gpu=1 --test-freq=10240 --print-freq=1024 --kl-multiplier=100.0
