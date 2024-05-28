#!/usr/bin/bash

python -u build_dataset.py \
--device cuda \
--config_path configs/YouTube.json \
--data_path datasets/youtube