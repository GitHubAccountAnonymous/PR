#!/usr/bin/bash
MODEL=EfficientPunct
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
--num_workers 16 \
--batch_size 256 \
--lr 1e-5 \
--epochs 3 \
--save_freq 50 \
--featurize