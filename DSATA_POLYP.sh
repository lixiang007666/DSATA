#!/bin/bash

#Please modify the following roots to yours.
dataset_root=datasets/Polyp
model_root=POLYP/models/segformer/
path_save_log=POLYP/logs

#Dataset [BKAI, CVC-ClinicDB, ETIS-LaribPolypDB, Kvasir-SEG]
Source=BKAI

#Optimizer
optimizer=Adam
lr=0.01

#Hyperparameters
memory_size=40
neighbor=16
prompt_alpha=0.01
warm_n=5
epoch=60

#Command
cd POLYP
CUDA_VISIBLE_DEVICES=0 python dsata.py \
--dataset_root $dataset_root --model_root $model_root --path_save_log $path_save_log \
--Source_Dataset $Source \
--optimizer $optimizer --lr $lr \
--memory_size $memory_size --neighbor $neighbor --prompt_alpha $prompt_alpha --warm_n $warm_n \
--epoch $epoch
