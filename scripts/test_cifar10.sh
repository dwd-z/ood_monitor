#!/usr/bin/env bash

METHOD=$1
OUT_DATA=$2

python test_ood.py \
--name test_${METHOD}_${OUT_DATA} \
--in_datadir dataset/id_data/CIFAR10 \
--out_datadir dataset/ood_data/${OUT_DATA} \
--model_name wrn \
--model WRN-40-2 \
--dropRate 0.3 \
--model_path checkpoints/cifar10_wrn_normal_standard_epoch_199.pt \
--batch 200 \
--logdir checkpoints/test_log \
--score ${METHOD} \
--epsilon_odin 0.0014 \
--input_size 32 \
--mean 0.492 0.482 0.446 \
--std 0.247 0.244 0.262 \
--num_ood 2000