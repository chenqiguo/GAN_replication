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
Which by default assumes your training set (images) is downloaded into the root folder ```data``` in this directory, and will prepare the cached HDF5 at 128x128 pixel resolution.

# 1.2 Metrics and Sampling
During training, this script will output logs with training metrics and test metrics, will save multiple copies (2 most recent and 5 highest-scoring) of the model weights/optimizer params, and will produce samples and interpolations every time it saves weights. The logs folder contains scripts to process these logs and plot the results using MATLAB (sorry not sorry).

After training, one can use ```sample.py``` to produce additional samples and interpolations, test with different truncation values, batch sizes, number of standing stat accumulations, etc. 

By default, everything is saved to weights/samples/logs/data folders.

# 1.3 An Important Note on Inception Metrics

This repo uses the PyTorch in-built inception network to calculate IS and FID. These scores are different from the scores you would get using the official TF inception code, and are only for monitoring purposes! Run sample.py on your model, with the --sample_npz argument, then run inception_tf13 to calculate the actual TensorFlow IS. Note that you will need to have TensorFlow 1.3 or earlier installed, as TF1.4+ breaks the original IS code.






