#!/bin/bash
# Same official-script hyperparams, but with the download.sh checkpoint (augreg IN21k, no IN1k ft)
export HF_HUB_OFFLINE=1
cd /root/autodl-tmp/Fly-CL
PY=/root/miniconda3/bin/python
M=vit_base_patch16_224_in21k

echo "[$(date)] === CIFAR-100 in21k ==="
$PY main.py --dataset CIFAR-100 --num_classes 100 --num_tasks 10 --model_name $M --embedding_dim 768 --expand_dim 10000 --synaptic_degree 300 --coding_level 0.3 --seed 1993 --batch_size 128 --gpu 0 --data_augmentation vit --ridge_lower 6 --ridge_upper 10 > /root/autodl-tmp/logs/cifar100_in21k.log 2>&1
echo "[$(date)] CIFAR-100 exit=$?"
echo "[$(date)] === CUB in21k ==="
$PY main.py --dataset CUB-200-2011 --num_classes 200 --num_tasks 10 --model_name $M --embedding_dim 768 --expand_dim 10000 --synaptic_degree 300 --coding_level 0.3 --seed 2023 --batch_size 128 --gpu 0 --data_augmentation vit --ridge_lower 6 --ridge_upper 10 > /root/autodl-tmp/logs/cub_in21k.log 2>&1
echo "[$(date)] CUB exit=$?"
echo "[$(date)] === VTAB in21k ==="
$PY main.py --dataset VTAB --num_classes 50 --num_tasks 5 --model_name $M --embedding_dim 768 --expand_dim 10000 --synaptic_degree 300 --coding_level 0.3 --seed 2023 --batch_size 128 --gpu 0 --data_augmentation vit --ridge_lower 6 --ridge_upper 10 > /root/autodl-tmp/logs/vtab_in21k.log 2>&1
echo "[$(date)] VTAB exit=$?"
echo "[$(date)] ALL DONE IN21K"
