#!/usr/bin/env bash

export NGPUS=8
srun --partition=Metric --mpi=pmi2 --gres=gpu:8 -n1 --ntasks-per-node=8 \
python -m torch.distributed.launch --nproc_per_node=$NGPUS train.py \
     --dataset citys --batch-size 8 --test-batch-size 2 --lr 0.0001 \
     --base-size 1024 --crop-size 768 \
     --epochs 240 -j 16 --warmup-factor 0.1 --warmup-iters 200 --log-step 10 --eval-epochs -1 --save-epoch 40

#python train.py \
#     --dataset citys --batch-size 2 --test-batch-size 2 --lr 0.0001 \
#     --base-size 1024 --crop-size 768 \
#     --epochs 240 -j 16 --warmup-factor 0.01 --log-step 10 --eval-epochs -1 --save-epoch 20