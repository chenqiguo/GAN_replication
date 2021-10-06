# GAN_replication
ICCV 2021 When does GAN replicate? An indication on the choice of dataset size

# 1. BigGAN-PyTorch
This contains code for GPU training of BigGANs from Large Scale GAN Training for High Fidelity Natural Image Synthesis by Andrew Brock, Jeff Donahue, and Karen Simonyan.

This code is by Andy Brock and Alex Andonian.

# 1.1 How To Use This Code
You will need:

PyTorch, version 1.0.1
tqdm, numpy, scipy, and h5py
The training set (for example, ImageNet)

First, you may optionally prepare a pre-processed HDF5 version of your target dataset for faster I/O. Following this (or not), you'll need the Inception moments needed to calculate FID. These can both be done by modifying and running:
```
./scripts/utils/prepare_data.sh
```
