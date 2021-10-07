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
This repo uses the PyTorch in-built inception network to calculate IS and FID. These scores are different from the scores you would get using the official TF inception code, and are only for monitoring purposes! Run sample.py on your model, with the ```--sample_npz``` argument, then run inception_tf13 to calculate the actual TensorFlow IS. Note that you will need to have TensorFlow 1.3 or earlier installed, as TF1.4+ breaks the original IS code.

# 1.4 1-Nearest Neighbor Query
Here we provide 1-NN query on the original training image for each GAN generated image in 3 different latent space.
(a) To run 1-NN query in pixel-wise space:
```
python NN_query_thresh_finalVer.py
```
(b) To run 1-NN query in inceptionV3 space: for example
```
python NNquery_inceptionv3_myTest.py \
 --dataset FLOWER_128_sub1000 --data_dir /Usr/BigGAN-PyTorch/imgs/FLOWER_128_sub1000/ \
 --gan_dir /Usr/gan_results_for_presentation/biggan/NNquery_inception_v3/FLOWER_128_sub1000/Itr38950/view_sampleSheetImgs/ \
 --result_dir /Usr/gan_results_for_presentation/biggan/NNquery_inception_v3/FLOWER_128_sub1000/Itr38950/ \
 --num_row 32 --num_col 32 
```
(c) To run 1-NN query in inceptionV3 concatenating pixel-wise space: for example
```
python NNquery_inceptionv3_pixelwise_myTest.py \
 --dataset FLOWER_128_sub1000 --data_dir /Usr/BigGAN-PyTorch/imgs/FLOWER_128_sub1000/ \
 --gan_dir /Usr/gan_results_for_presentation/biggan/NNquery_inceptionv3_pixelwise/FLOWER_128_sub1000/Itr38950/view_sampleSheetImgs/ \
 --result_dir /Usr/gan_results_for_presentation/biggan/NNquery_inceptionv3_pixelwise/FLOWER_128_sub1000/Itr38950/ \
 --num_row 32 --num_col 32
 ```

