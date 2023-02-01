#!/bin/sh

SAVE_LOC=$1
DATA_LOC=$2
EPOCHS=$3
CEPOCHS=$4
BATCH_SIZE=$5
LR=$6
CLR=$7

python train.py \
-train 1 \
-train_ckpt $DATA_LOC \
-save_location $SAVE_LOC \
-learning_rate $LR \
-train_batch_size $BATCH_SIZE \
-epoch $EPOCHS

python train.py \
-train 1 \
-c_train 1 \
-saved_ckpt $EPOCHS \
-train_ckpt $DATA_LOC \
-save_location $SAVE_LOC \
-train_batch_size $BATCH_SIZE \
-learning_rate $CLR \
-epoch $CEPOCHS

