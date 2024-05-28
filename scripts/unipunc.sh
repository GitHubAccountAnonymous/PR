#!/usr/bin/bash
MODEL=UniPunc
DATASET=youtube
DEVICES=0
export CUDA_VISIBLE_DEVICES=${DEVICES}

python -u main.py \
--device cpu \
--config_path configs/models/$MODEL.json \
--data_path datasets/youtube_trial \
--save_path models/trained/$DATASET/$MODEL \
--mode train \
--optimizer adamw \
--num_workers 2 \
--batch_size 16 \
--lr 1e-5 \
--epochs 2 \
--save_freq 10