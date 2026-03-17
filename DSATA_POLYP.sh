#!/bin/bash

#Please modify the following roots to yours.
dataset_root=datasets/Polyp
model_root=POLYP/models/
path_save_log=POLYP/logs

#Dataset [BKAI, CVC-ClinicDB, ETIS-LaribPolypDB, Kvasir-SEG]
Source=BKAI

#Optimizer
optimizer=Adam
lr=0.01

#Command
cd POLYP
CUDA_VISIBLE_DEVICES=0 python dsata.py \
--dataset_root $dataset_root --model_root $model_root --path_save_log $path_save_log \
--Source_Dataset $Source \
--optimizer $optimizer --lr $lr
