#!/bin/bash
./bench/tensorized_dlrm.sh --load-model='saved_models/CP' --tensor-type='CP' --use-gpu=0 --test-freq=248 --print-freq=124 --kl-multiplier=1.0