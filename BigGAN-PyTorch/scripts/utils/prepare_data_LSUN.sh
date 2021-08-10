#!/bin/bash
python make_hdf5.py --dataset LSUN_bedroom --batch_size 64 --data_root data
# python calculate_inception_moments.py --dataset I128_hdf5 --data_root data