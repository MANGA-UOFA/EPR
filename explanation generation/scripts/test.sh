#!/bin/sh

SAVE_LOC=$1
DATA_LOC=$2
EVAL_BATCH=$3
RUN_CKPT=$4

python train.py \
-train 0 \
-train_ckpt $DATA_LOC \
-save_location $SAVE_LOC \
-eval_batch_size $EVAL_BATCH \
-saved_ckpt $RUN_CKPT

