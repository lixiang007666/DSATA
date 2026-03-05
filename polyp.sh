#!/bin/bash

# 创建日志目录
echo "开始在所有4个GPU上并行运行训练任务..."
script=POLYP/dsata.py
# 使用并行后台执行
CUDA_VISIBLE_DEVICES=0 python $script --dataset_root datasets/Polyp --model_root POLYP/models/segformer/ --path_save_log POLYP/logs --Source_Dataset BKAI --optimizer Adam --lr 0.01 --memory_size 40 --neighbor 16 --prompt_alpha 0.01 --warm_n 5 --epoch 60 

CUDA_VISIBLE_DEVICES=1 python $script --dataset_root datasets/Polyp --model_root POLYP/models/segformer/ --path_save_log POLYP/logs --Source_Dataset CVC-ClinicDB --optimizer Adam --lr 0.01 --memory_size 40 --neighbor 16 --prompt_alpha 0.01 --warm_n 5 --epoch 130 

CUDA_VISIBLE_DEVICES=2 python $script --dataset_root datasets/Polyp --model_root POLYP/models/segformer/ --path_save_log POLYP/logs --Source_Dataset ETIS-LaribPolypDB --optimizer Adam --lr 0.01 --memory_size 40 --neighbor 16 --prompt_alpha 0.01 --warm_n 5 --epoch 90 

CUDA_VISIBLE_DEVICES=3 python $script --dataset_root datasets/Polyp --model_root POLYP/models/segformer/ --path_save_log POLYP/logs --Source_Dataset Kvasir-SEG --optimizer Adam --lr 0.01 --memory_size 40 --neighbor 16 --prompt_alpha 0.01 --warm_n 5 --epoch 90 



echo "所有训练任务已完成！"