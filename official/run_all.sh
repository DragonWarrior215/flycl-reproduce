#!/bin/bash
# Reproduce Fly-CL on CIFAR-100 / CUB-200-2011 / VTAB with the official repo.
# Args identical to scripts/test_{cifar,cub,vtab}.sh except --gpu 0 (single-GPU server).
export HF_HUB_OFFLINE=1 HF_ENDPOINT=https://hf-mirror.com
cd /root/autodl-tmp/Fly-CL
PY=/root/miniconda3/bin/python

echo "[$(date)] === CIFAR-100 ==="
$PY main.py --dataset CIFAR-100 --num_classes 100 --num_tasks 10 --model_name vit_base_patch16_224 --embedding_dim 768 --expand_dim 10000 --synaptic_degree 300 --coding_level 0.3 --seed 1993 --batch_size 128 --gpu 0 --data_augmentation vit --ridge_lower 6 --ridge_upper 10 > /root/autodl-tmp/logs/cifar100.log 2>&1
echo "[$(date)] CIFAR-100 exit=$?"

echo "[$(date)] === CUB-200-2011 ==="
$PY main.py --dataset CUB-200-2011 --num_classes 200 --num_tasks 10 --model_name vit_base_patch16_224 --embedding_dim 768 --expand_dim 10000 --synaptic_degree 300 --coding_level 0.3 --seed 2023 --batch_size 128 --gpu 0 --data_augmentation vit --ridge_lower 6 --ridge_upper 10 > /root/autodl-tmp/logs/cub.log 2>&1
echo "[$(date)] CUB exit=$?"

echo "[$(date)] === VTAB ==="
$PY main.py --dataset VTAB --num_classes 50 --num_tasks 5 --model_name vit_base_patch16_224 --embedding_dim 768 --expand_dim 10000 --synaptic_degree 300 --coding_level 0.3 --seed 2023 --batch_size 128 --gpu 0 --data_augmentation vit --ridge_lower 6 --ridge_upper 10 > /root/autodl-tmp/logs/vtab.log 2>&1
echo "[$(date)] VTAB exit=$?"
echo "[$(date)] ALL DONE"
