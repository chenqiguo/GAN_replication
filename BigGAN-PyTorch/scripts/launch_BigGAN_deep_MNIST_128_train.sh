#!/bin/bash
#--batch_size 32
python train.py \
--model BigGANdeep --data_root /eecf/cbcsl/data100b/Chenqi/data/MNIST/resized \
--dataset MNIST_128_train --shuffle --num_workers 16 --batch_size 16 \
--device '0' \
--num_G_accumulations 64 --num_D_accumulations 64 \
--num_D_steps 1 --G_lr 1e-4 --D_lr 4e-4 --D_B2 0.999 --G_B2 0.999 \
--G_attn 64 --D_attn 64 \
--G_ch 128 --D_ch 128 \
--G_depth 2 --D_depth 2 \
--G_nl inplace_relu --D_nl inplace_relu \
--SN_eps 1e-6 --BN_eps 1e-5 --adam_eps 1e-6 \
--G_ortho 0.0 \
--G_shared \
--G_init ortho --D_init ortho \
--hier --dim_z 128 --shared_dim 128 \
--ema --ema_decay 0.999 --use_ema --ema_start 15000 --G_eval_mode \
--test_every 200 --save_every 50 --num_best_copies 5 --num_save_copies 5 --seed 0 \
--use_multiepoch_sampler \
--num_epochs 380 --resume \
#--num_epochs 350 #400