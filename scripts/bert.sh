#!/usr/bin/bash
MODEL=BERTMLP
DATASET=mustcv1
DEVICES=0
export CUDA_VISIBLE_DEVICES=${DEVICES}

python -u main.py \
--device cuda \
--config_path configs/models/$MODEL.json \
--data_path datasets/$DATASET \
--save_path models/trained/$DATASET/$MODEL \
--mode train \
--optimizer adamw \
--num_workers 1 \
--batch_size 1 \
--lr 1e-5 \
--epochs 10 \
--save_freq 1000 \
--featurize
