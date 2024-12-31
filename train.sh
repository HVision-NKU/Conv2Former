#!/usr/bin/env bash
MODEL=$1
BATCH_SIZE=$2 
bash distributed_train.sh 8 /path/to/imagenet \
	  --model $MODEL -b 128 --lr 1e-3 --drop-path $DROP_PATH \