#!/bin/bash

#Please modify the following roots to yours.
dataset_root=datasets/Fundus
model_root=OPTIC/models/
path_save_log=OPTIC/logs

#Dataset [RIM_ONE_r3, REFUGE, ORIGA, REFUGE_Valid, Drishti_GS]
Source=RIM_ONE_r3

#Optimizer
optimizer=Adam
lr=0.005

#Command
cd Fundus
CUDA_VISIBLE_DEVICES=0 python dsata.py \
--dataset_root $dataset_root --model_root $model_root --path_save_log $path_save_log \
--Source_Dataset $Source \
--optimizer $optimizer --lr $lr
