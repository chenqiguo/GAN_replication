#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 12:41:21 2020

@author: guo.1648
"""

# referenced from NN_query_testCode_forBiggan.py.

# this code is for biggan sample sheet.

# this code also do 1 NN matching for the generated images and the original images,
# but the purpose is to compute the corresponding matching distance for each result,
# and then use these matching distances and human perception for each mathing pair,
# to find out the NN distance threshold which is the largest matching distance satisfying
# 100% perceptual replication.

# Note: we will combine this biggan result and the stylegan2 result later to find out the threshold!


import cv2
import os
import numpy as np
#from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestNeighbors
#from skimage import img_as_ubyte
#import torchvision.transforms as transforms

"""
#### for FLOWER_128_sub1000: 1000 images dataset
src_sampleSheetImg = '/scratch/BigGAN-PyTorch/samples/BigGAN_FLOWER_128_sub1000_BigGANdeep_seed0_Gch128_Dch128_Gd2_Dd2_bs16_nDa64_nGa64_Glr1.0e-04_Dlr4.0e-04_Gnlinplace_relu_Dnlinplace_relu_Ginitortho_Dinitortho_Gattn64_Dattn64_Gshared_hier_ema/fixed_samples38800.jpg'
srcRootDir_originDataImg = '/scratch/BigGAN-PyTorch/imgs/flower_sub1000/'

dstRootDir_viewSampleSheetImgs = '/scratch/BigGAN-PyTorch/imgs/NN_query/FLOWER_128_sub1000/samples38800/view_sampleSheetImgs/'
dstRootDir_NNmatchResult = '/scratch/BigGAN-PyTorch/imgs/NN_query/FLOWER_128_sub1000/samples38800/NNmatchResult/'
dstImgName_NNmatchSheet = '/scratch/BigGAN-PyTorch/imgs/NN_query/FLOWER_128_sub1000/samples38800/NNmatchResultSheet.jpg'
dstTxtName_matchDist = '/scratch/BigGAN-PyTorch/imgs/NN_query/FLOWER_128_sub1000/samples38800/NNmatchDist.txt'

# parameters:
im_size = 128
batch_size = 16 # i.e., the sample sheet is of 4x4 !!!!:
num_row = 4
num_col = 4
"""
"""
#### for FLOWER_128: 8189 images dataset (the original FLOWER dataset)
src_sampleSheetImg = '/scratch/BigGAN-PyTorch/samples/BigGAN_FLOWER_128_BigGANdeep_seed0_Gch128_Dch128_Gd2_Dd2_bs48_nDa64_nGa64_Glr1.0e-04_Dlr4.0e-04_Gnlinplace_relu_Dnlinplace_relu_Ginitortho_Dinitortho_Gattn64_Dattn64_Gshared_hier_ema/fixed_samples21400.jpg'
srcRootDir_originDataImg = '/scratch/BigGAN-PyTorch/data/flower/jpg/'

dstRootDir_viewSampleSheetImgs = '/scratch/BigGAN-PyTorch/imgs/NN_query/FLOWER_128/samples21400/view_sampleSheetImgs/'
dstRootDir_NNmatchResult = '/scratch/BigGAN-PyTorch/imgs/NN_query/FLOWER_128/samples21400/NNmatchResult/'
dstImgName_NNmatchSheet = '/scratch/BigGAN-PyTorch/imgs/NN_query/FLOWER_128/samples21400/NNmatchResultSheet.jpg'
dstTxtName_matchDist = '/scratch/BigGAN-PyTorch/imgs/NN_query/FLOWER_128/samples21400/NNmatchDist.txt'

# parameters:
im_size = 128
batch_size = 48 # i.e., the sample sheet is of 6x8 !!!!:
num_row = 8
num_col = 6
"""
"""
#### for FLOWER_128_sub4000: 4000 images dataset
src_sampleSheetImg = '/scratch/BigGAN-PyTorch/samples/BigGAN_FLOWER_128_sub4000_BigGANdeep_seed0_Gch128_Dch128_Gd2_Dd2_bs32_nDa64_nGa64_Glr1.0e-04_Dlr4.0e-04_Gnlinplace_relu_Dnlinplace_relu_Ginitortho_Dinitortho_Gattn64_Dattn64_Gshared_hier_ema/fixed_samples8800.jpg'
# newly modified: different from that in FLOWER_128_sub1000 and FLOWER_128:
#srcRootDir_originDataImg = '/scratch/BigGAN-PyTorch/imgs/flower_sub4000/'
srcRootDir_imgNpz = '/scratch/BigGAN-PyTorch/FLOWER_128_sub4000_imgs.npz'

dstRootDir_viewSampleSheetImgs = '/scratch/BigGAN-PyTorch/imgs/NN_query/FLOWER_128_sub4000/samples8800/view_sampleSheetImgs/'
dstRootDir_NNmatchResult = '/scratch/BigGAN-PyTorch/imgs/NN_query/FLOWER_128_sub4000/samples8800/NNmatchResult/'
dstImgName_NNmatchSheet = '/scratch/BigGAN-PyTorch/imgs/NN_query/FLOWER_128_sub4000/samples8800/NNmatchResultSheet.jpg'
dstTxtName_matchDist = '/scratch/BigGAN-PyTorch/imgs/NN_query/FLOWER_128_sub4000/samples8800/NNmatchDist.txt'

# parameters:
im_size = 128
batch_size = 32 # i.e., the sample sheet is of 5x6+2 !!!!:
num_row = 7
num_col = 5
"""
"""
#### for CelebA_128_sub1000: 1000 images dataset
src_sampleSheetImg = '/scratch/BigGAN-PyTorch/samples/BigGAN_CelebA_128_sub1000_BigGANdeep_seed0_Gch128_Dch128_Gd2_Dd2_bs16_nDa64_nGa64_Glr1.0e-04_Dlr4.0e-04_Gnlinplace_relu_Dnlinplace_relu_Ginitortho_Dinitortho_Gattn64_Dattn64_Gshared_hier_ema/fixed_samples27000.jpg'
# newly modified: different from that in FLOWER_128_sub1000 and FLOWER_128:
#srcRootDir_originDataImg = '/scratch/BigGAN-PyTorch/imgs/flower_sub4000/'
srcRootDir_imgNpz = '/scratch/BigGAN-PyTorch/CelebA_128_sub1000_imgs.npz'

dstRootDir_viewSampleSheetImgs = '/scratch/BigGAN-PyTorch/imgs/NN_query/CelebA_128_sub1000/samples27000/view_sampleSheetImgs/'
dstRootDir_NNmatchResult = '/scratch/BigGAN-PyTorch/imgs/NN_query/CelebA_128_sub1000/samples27000/NNmatchResult/'
dstImgName_NNmatchSheet = '/scratch/BigGAN-PyTorch/imgs/NN_query/CelebA_128_sub1000/samples27000/NNmatchResultSheet.jpg'
dstTxtName_matchDist = '/scratch/BigGAN-PyTorch/imgs/NN_query/CelebA_128_sub1000/samples27000/NNmatchDist.txt'

# parameters:
im_size = 128
batch_size = 16 # i.e., the sample sheet is of 4x4 !!!!:
num_row = 4
num_col = 4
"""
"""
#### for CelebA_128_sub4000: 4000 images dataset
src_sampleSheetImg = '/eecf/cbcsl/data100b/Chenqi/BigGAN-PyTorch/samples/BigGAN_CelebA_128_sub4000_BigGANdeep_seed0_Gch128_Dch128_Gd2_Dd2_bs32_nDa64_nGa64_Glr1.0e-04_Dlr4.0e-04_Gnlinplace_relu_Dnlinplace_relu_Ginitortho_Dinitortho_Gattn64_Dattn64_Gshared_hier_ema/fixed_samples16200.jpg'
# newly modified: different from that in FLOWER_128_sub1000 and FLOWER_128:
#srcRootDir_originDataImg = '/scratch/BigGAN-PyTorch/imgs/flower_sub4000/'
srcRootDir_imgNpz = '/eecf/cbcsl/data100b/Chenqi/BigGAN-PyTorch/CelebA_128_sub4000_imgs.npz'

dstRootDir_viewSampleSheetImgs = '/eecf/cbcsl/data100b/Chenqi/BigGAN-PyTorch/imgs/NN_query/CelebA_128_sub4000/samples16200/view_sampleSheetImgs/'
dstRootDir_NNmatchResult = '/eecf/cbcsl/data100b/Chenqi/BigGAN-PyTorch/imgs/NN_query/CelebA_128_sub4000/samples16200/NNmatchResult/'
dstImgName_NNmatchSheet = '/eecf/cbcsl/data100b/Chenqi/BigGAN-PyTorch/imgs/NN_query/CelebA_128_sub4000/samples16200/NNmatchResultSheet.jpg'
dstTxtName_matchDist = '/eecf/cbcsl/data100b/Chenqi/BigGAN-PyTorch/imgs/NN_query/CelebA_128_sub4000/samples16200/NNmatchDist.txt'

# parameters:
im_size = 128
batch_size = 32 # i.e., the sample sheet is of 5x6+2 !!!!:
num_row = 7
num_col = 5
"""
"""
#### for CelebA_128_sub8000: 8000 images dataset
src_sampleSheetImg = '/eecf/cbcsl/data100b/Chenqi/BigGAN-PyTorch/samples/BigGAN_CelebA_128_sub8000_BigGANdeep_seed0_Gch128_Dch128_Gd2_Dd2_bs32_nDa64_nGa64_Glr1.0e-04_Dlr4.0e-04_Gnlinplace_relu_Dnlinplace_relu_Ginitortho_Dinitortho_Gattn64_Dattn64_Gshared_hier_ema/fixed_samples20400.jpg'
# newly modified: different from that in FLOWER_128_sub1000 and FLOWER_128:
#srcRootDir_originDataImg = '/scratch/BigGAN-PyTorch/imgs/flower_sub4000/'
srcRootDir_imgNpz = '/eecf/cbcsl/data100b/Chenqi/BigGAN-PyTorch/CelebA_128_sub8000_imgs.npz'

dstRootDir_viewSampleSheetImgs = '/eecf/cbcsl/data100b/Chenqi/BigGAN-PyTorch/imgs/NN_query/CelebA_128_sub8000/samples20400/view_sampleSheetImgs/'
dstRootDir_NNmatchResult = '/eecf/cbcsl/data100b/Chenqi/BigGAN-PyTorch/imgs/NN_query/CelebA_128_sub8000/samples20400/NNmatchResult/'
dstImgName_NNmatchSheet = '/eecf/cbcsl/data100b/Chenqi/BigGAN-PyTorch/imgs/NN_query/CelebA_128_sub8000/samples20400/NNmatchResultSheet.jpg'
dstTxtName_matchDist = '/eecf/cbcsl/data100b/Chenqi/BigGAN-PyTorch/imgs/NN_query/CelebA_128_sub8000/samples20400/NNmatchDist.txt'

# parameters:
im_size = 128
batch_size = 32 # i.e., the sample sheet is of 5x6+2 !!!!:
num_row = 7
num_col = 5
"""
"""
#### for CelebA_128_sub200: 200 images dataset
src_sampleSheetImg = '/eecf/cbcsl/data100b/Chenqi/BigGAN-PyTorch/samples/BigGAN_CelebA_128_sub200_BigGANdeep_seed0_Gch128_Dch128_Gd2_Dd2_bs32_nDa64_nGa64_Glr1.0e-04_Dlr4.0e-04_Gnlinplace_relu_Dnlinplace_relu_Ginitortho_Dinitortho_Gattn64_Dattn64_Gshared_hier_ema/fixed_samples17800.jpg'
# newly modified: different from that in FLOWER_128_sub1000 and FLOWER_128:
#srcRootDir_originDataImg = '/scratch/BigGAN-PyTorch/imgs/flower_sub4000/'
srcRootDir_imgNpz = '/eecf/cbcsl/data100b/Chenqi/BigGAN-PyTorch/CelebA_128_sub200_imgs.npz'

dstRootDir_viewSampleSheetImgs = '/eecf/cbcsl/data100b/Chenqi/BigGAN-PyTorch/imgs/NN_query/CelebA_128_sub200/samples17800/view_sampleSheetImgs/'
dstRootDir_NNmatchResult = '/eecf/cbcsl/data100b/Chenqi/BigGAN-PyTorch/imgs/NN_query/CelebA_128_sub200/samples17800/NNmatchResult/'
dstImgName_NNmatchSheet = '/eecf/cbcsl/data100b/Chenqi/BigGAN-PyTorch/imgs/NN_query/CelebA_128_sub200/samples17800/NNmatchResultSheet.jpg'
dstTxtName_matchDist = '/eecf/cbcsl/data100b/Chenqi/BigGAN-PyTorch/imgs/NN_query/CelebA_128_sub200/samples17800/NNmatchDist.txt'

# parameters:
im_size = 128
batch_size = 32 # i.e., the sample sheet is of 5x6+2 !!!!:
num_row = 7
num_col = 5
"""
"""
#### for FLOWER_128_sub4000_bs56: 4000 images dataset
src_sampleSheetImg = '/eecf/cbcsl/data100b/Chenqi/BigGAN-PyTorch/samples/BigGAN_FLOWER_128_sub4000_BigGANdeep_seed0_Gch128_Dch128_Gd2_Dd2_bs56_nDa64_nGa64_Glr1.0e-04_Dlr4.0e-04_Gnlinplace_relu_Dnlinplace_relu_Ginitortho_Dinitortho_Gattn64_Dattn64_Gshared_hier_ema/fixed_samples10000.jpg'
# newly modified: different from that in FLOWER_128_sub1000 and FLOWER_128:
#srcRootDir_originDataImg = '/scratch/BigGAN-PyTorch/imgs/flower_sub4000/'
srcRootDir_imgNpz = '/eecf/cbcsl/data100b/Chenqi/BigGAN-PyTorch/FLOWER_128_sub4000_imgs.npz'

dstRootDir_viewSampleSheetImgs = '/eecf/cbcsl/data100b/Chenqi/BigGAN-PyTorch/imgs/NN_query/FLOWER_128_sub4000_bs56/samples10000/view_sampleSheetImgs/'
dstRootDir_NNmatchResult = '/eecf/cbcsl/data100b/Chenqi/BigGAN-PyTorch/imgs/NN_query/FLOWER_128_sub4000_bs56/samples10000/NNmatchResult/'
dstImgName_NNmatchSheet = '/eecf/cbcsl/data100b/Chenqi/BigGAN-PyTorch/imgs/NN_query/FLOWER_128_sub4000_bs56/samples10000/NNmatchResultSheet.jpg'
dstTxtName_matchDist = '/eecf/cbcsl/data100b/Chenqi/BigGAN-PyTorch/imgs/NN_query/FLOWER_128_sub4000_bs56/samples10000/NNmatchDist.txt'

# parameters:
im_size = 128
batch_size = 56 # i.e., the sample sheet is of 8x7 !!!!:
num_row = 8
num_col = 7
"""
"""
#### for FLOWER_128_sub6000: 6000 images dataset
src_sampleSheetImg = '/eecf/cbcsl/data100b/Chenqi/BigGAN-PyTorch/samples/BigGAN_FLOWER_128_sub6000_BigGANdeep_seed0_Gch128_Dch128_Gd2_Dd2_bs24_nDa64_nGa64_Glr1.0e-04_Dlr4.0e-04_Gnlinplace_relu_Dnlinplace_relu_Ginitortho_Dinitortho_Gattn64_Dattn64_Gshared_hier_ema/fixed_samples17000.jpg'
# newly modified: different from that in FLOWER_128_sub1000 and FLOWER_128:
#srcRootDir_originDataImg = '/scratch/BigGAN-PyTorch/imgs/flower_sub4000/'
srcRootDir_imgNpz = '/eecf/cbcsl/data100b/Chenqi/BigGAN-PyTorch/FLOWER_128_sub6000_imgs.npz'

dstRootDir_viewSampleSheetImgs = '/eecf/cbcsl/data100b/Chenqi/BigGAN-PyTorch/imgs/NN_query/FLOWER_128_sub6000/samples17000/view_sampleSheetImgs/'
dstRootDir_NNmatchResult = '/eecf/cbcsl/data100b/Chenqi/BigGAN-PyTorch/imgs/NN_query/FLOWER_128_sub6000/samples17000/NNmatchResult/'
dstImgName_NNmatchSheet = '/eecf/cbcsl/data100b/Chenqi/BigGAN-PyTorch/imgs/NN_query/FLOWER_128_sub6000/samples17000/NNmatchResultSheet.jpg'
dstTxtName_matchDist = '/eecf/cbcsl/data100b/Chenqi/BigGAN-PyTorch/imgs/NN_query/FLOWER_128_sub6000/samples17000/NNmatchDist.txt'

# parameters:
im_size = 128
batch_size = 24 # i.e., the sample sheet is of 6x4 !!!!:
num_row = 6
num_col = 4
"""
"""
#### for MNIST_128_train: 60000 images dataset
src_sampleSheetImg = '/eecf/cbcsl/data100b/Chenqi/BigGAN-PyTorch/samples/BigGAN_MNIST_128_train_BigGANdeep_seed0_Gch128_Dch128_Gd2_Dd2_bs16_nDa64_nGa64_Glr1.0e-04_Dlr4.0e-04_Gnlinplace_relu_Dnlinplace_relu_Ginitortho_Dinitortho_Gattn64_Dattn64_Gshared_hier_ema/fixed_samples34400.jpg'
# newly modified: different from that in FLOWER_128_sub1000 and FLOWER_128:
#srcRootDir_originDataImg = '/scratch/BigGAN-PyTorch/imgs/flower_sub4000/'
srcRootDir_imgNpz = '/eecf/cbcsl/data100b/Chenqi/BigGAN-PyTorch/MNIST_128_train_imgs.npz'

dstRootDir_viewSampleSheetImgs = '/eecf/cbcsl/data100b/Chenqi/BigGAN-PyTorch/imgs/NN_query/MNIST_128_train/samples34400/view_sampleSheetImgs/'
dstRootDir_NNmatchResult = '/eecf/cbcsl/data100b/Chenqi/BigGAN-PyTorch/imgs/NN_query/MNIST_128_train/samples34400/NNmatchResult/'
dstImgName_NNmatchSheet = '/eecf/cbcsl/data100b/Chenqi/BigGAN-PyTorch/imgs/NN_query/MNIST_128_train/samples34400/NNmatchResultSheet.jpg'
dstTxtName_matchDist = '/eecf/cbcsl/data100b/Chenqi/BigGAN-PyTorch/imgs/NN_query/MNIST_128_train/samples34400/NNmatchDist.txt'

# parameters:
im_size = 128
batch_size = 16 # i.e., the sample sheet is of 4x4 !!!!:
num_row = 4
num_col = 4
"""
"""
#### for CelebA_128_sub600: 600 images dataset
src_sampleSheetImg = '/eecf/cbcsl/data100b/Chenqi/BigGAN-PyTorch/samples/BigGAN_CelebA_128_sub600_BigGANdeep_seed0_Gch128_Dch128_Gd2_Dd2_bs32_nDa64_nGa64_Glr1.0e-04_Dlr4.0e-04_Gnlinplace_relu_Dnlinplace_relu_Ginitortho_Dinitortho_Gattn64_Dattn64_Gshared_hier_ema/fixed_samples17400.jpg'
# newly modified: different from that in FLOWER_128_sub1000 and FLOWER_128:
#srcRootDir_originDataImg = '/scratch/BigGAN-PyTorch/imgs/flower_sub4000/'
srcRootDir_imgNpz = '/eecf/cbcsl/data100b/Chenqi/BigGAN-PyTorch/CelebA_128_sub600_imgs.npz'

dstRootDir_viewSampleSheetImgs = '/eecf/cbcsl/data100b/Chenqi/BigGAN-PyTorch/imgs/NN_query/CelebA_128_sub600/samples17400/view_sampleSheetImgs/'
dstRootDir_NNmatchResult = '/eecf/cbcsl/data100b/Chenqi/BigGAN-PyTorch/imgs/NN_query/CelebA_128_sub600/samples17400/NNmatchResult/'
dstImgName_NNmatchSheet = '/eecf/cbcsl/data100b/Chenqi/BigGAN-PyTorch/imgs/NN_query/CelebA_128_sub600/samples17400/NNmatchResultSheet.jpg'
dstTxtName_matchDist = '/eecf/cbcsl/data100b/Chenqi/BigGAN-PyTorch/imgs/NN_query/CelebA_128_sub600/samples17400/NNmatchDist.txt'

# parameters:
im_size = 128
batch_size = 32 # i.e., the sample sheet is of 5x6+2 !!!!:
num_row = 7
num_col = 5
"""
"""
#### for FLOWER_128_sub2000: 2000 images dataset
src_sampleSheetImg = '/eecf/cbcsl/data100b/Chenqi/BigGAN-PyTorch/samples/BigGAN_FLOWER_128_sub2000_BigGANdeep_seed0_Gch128_Dch128_Gd2_Dd2_bs32_nDa64_nGa64_Glr1.0e-04_Dlr4.0e-04_Gnlinplace_relu_Dnlinplace_relu_Ginitortho_Dinitortho_Gattn64_Dattn64_Gshared_hier_ema/fixed_samples24600.jpg'
# newly modified: different from that in FLOWER_128_sub1000 and FLOWER_128:
#srcRootDir_originDataImg = '/scratch/BigGAN-PyTorch/imgs/flower_sub4000/'
srcRootDir_imgNpz = '/eecf/cbcsl/data100b/Chenqi/BigGAN-PyTorch/FLOWER_128_sub2000_imgs.npz'

dstRootDir_viewSampleSheetImgs = '/eecf/cbcsl/data100b/Chenqi/BigGAN-PyTorch/imgs/NN_query/FLOWER_128_sub2000/samples24600/view_sampleSheetImgs/'
dstRootDir_NNmatchResult = '/eecf/cbcsl/data100b/Chenqi/BigGAN-PyTorch/imgs/NN_query/FLOWER_128_sub2000/samples24600/NNmatchResult/'
dstImgName_NNmatchSheet = '/eecf/cbcsl/data100b/Chenqi/BigGAN-PyTorch/imgs/NN_query/FLOWER_128_sub2000/samples24600/NNmatchResultSheet.jpg'
dstTxtName_matchDist = '/eecf/cbcsl/data100b/Chenqi/BigGAN-PyTorch/imgs/NN_query/FLOWER_128_sub2000/samples24600/NNmatchDist.txt'

# parameters:
im_size = 128
batch_size = 32 # i.e., the sample sheet is of 5x6+2 !!!!:
num_row = 7
num_col = 5
"""
"""
#### for MNIST_128_sub10000: 10000 images dataset
src_sampleSheetImg = '/eecf/cbcsl/data100b/Chenqi/BigGAN-PyTorch/samples/BigGAN_MNIST_128_sub10000_BigGANdeep_seed0_Gch128_Dch128_Gd2_Dd2_bs16_nDa64_nGa64_Glr1.0e-04_Dlr4.0e-04_Gnlinplace_relu_Dnlinplace_relu_Ginitortho_Dinitortho_Gattn64_Dattn64_Gshared_hier_ema/fixed_samples34250.jpg'
# newly modified: different from that in FLOWER_128_sub1000 and FLOWER_128:
#srcRootDir_originDataImg = '/scratch/BigGAN-PyTorch/imgs/flower_sub4000/'
srcRootDir_imgNpz = '/eecf/cbcsl/data100b/Chenqi/BigGAN-PyTorch/MNIST_128_sub10000_imgs.npz'

dstRootDir_viewSampleSheetImgs = '/eecf/cbcsl/data100b/Chenqi/BigGAN-PyTorch/imgs/NN_query/MNIST_128_sub10000/samples34250/view_sampleSheetImgs/'
dstRootDir_NNmatchResult = '/eecf/cbcsl/data100b/Chenqi/BigGAN-PyTorch/imgs/NN_query/MNIST_128_sub10000/samples34250/NNmatchResult/'
dstImgName_NNmatchSheet = '/eecf/cbcsl/data100b/Chenqi/BigGAN-PyTorch/imgs/NN_query/MNIST_128_sub10000/samples34250/NNmatchResultSheet.jpg'
dstTxtName_matchDist = '/eecf/cbcsl/data100b/Chenqi/BigGAN-PyTorch/imgs/NN_query/MNIST_128_sub10000/samples34250/NNmatchDist.txt'

# parameters:
im_size = 128
batch_size = 16 # i.e., the sample sheet is of 4x4 !!!!:
num_row = 4
num_col = 4
"""
"""
#### for MNIST_128_sub30000: 30000 images dataset
src_sampleSheetImg = '/eecf/cbcsl/data100b/Chenqi/BigGAN-PyTorch/imgs/NN_query/MNIST_128_sub30000/samples32000/fixed_samples32000.jpg'
# newly modified: different from that in FLOWER_128_sub1000 and FLOWER_128:
#srcRootDir_originDataImg = '/scratch/BigGAN-PyTorch/imgs/flower_sub4000/'
srcRootDir_imgNpz = '/eecf/cbcsl/data100b/Chenqi/BigGAN-PyTorch/MNIST_128_sub30000_imgs.npz'

dstRootDir_viewSampleSheetImgs = '/eecf/cbcsl/data100b/Chenqi/BigGAN-PyTorch/imgs/NN_query/MNIST_128_sub30000/samples32000/view_sampleSheetImgs/'
dstRootDir_NNmatchResult = '/eecf/cbcsl/data100b/Chenqi/BigGAN-PyTorch/imgs/NN_query/MNIST_128_sub30000/samples32000/NNmatchResult/'
dstImgName_NNmatchSheet = '/eecf/cbcsl/data100b/Chenqi/BigGAN-PyTorch/imgs/NN_query/MNIST_128_sub30000/samples32000/NNmatchResultSheet.jpg'
dstTxtName_matchDist = '/eecf/cbcsl/data100b/Chenqi/BigGAN-PyTorch/imgs/NN_query/MNIST_128_sub30000/samples32000/NNmatchDist.txt'

# parameters:
im_size = 128
batch_size = 16 # i.e., the sample sheet is of 4x4 !!!!:
num_row = 4
num_col = 4
"""

#### for LSUN_128_sub200: 200 images dataset
src_sampleSheetImg = '/eecf/cbcsl/data100b/Chenqi/BigGAN-PyTorch/imgs/NN_query/LSUN_128_sub200/samples11800/fixed_samples11800.jpg'
# newly modified: different from that in FLOWER_128_sub1000 and FLOWER_128:
#srcRootDir_originDataImg = '/scratch/BigGAN-PyTorch/imgs/flower_sub4000/'
srcRootDir_imgNpz = '/eecf/cbcsl/data100b/Chenqi/BigGAN-PyTorch/LSUN_128_sub200_imgs.npz'

dstRootDir_viewSampleSheetImgs = '/eecf/cbcsl/data100b/Chenqi/BigGAN-PyTorch/imgs/NN_query/LSUN_128_sub200/samples11800/view_sampleSheetImgs/'
dstRootDir_NNmatchResult = '/eecf/cbcsl/data100b/Chenqi/BigGAN-PyTorch/imgs/NN_query/LSUN_128_sub200/samples11800/NNmatchResult/'
dstImgName_NNmatchSheet = '/eecf/cbcsl/data100b/Chenqi/BigGAN-PyTorch/imgs/NN_query/LSUN_128_sub200/samples11800/NNmatchResultSheet.jpg'
dstTxtName_matchDist = '/eecf/cbcsl/data100b/Chenqi/BigGAN-PyTorch/imgs/NN_query/LSUN_128_sub200/samples11800/NNmatchDist.txt'

# parameters:
im_size = 128
batch_size = 48 # i.e., the sample sheet is of 8x6 !!!!:
num_row = 8
num_col = 6


def dealWith_sampleSheet():
    sampleSheet_img = cv2.imread(src_sampleSheetImg)
    (sheet_img_height, sheet_img_width, ch) = sampleSheet_img.shape
    
    single_img_height = sheet_img_height//num_row # 130
    single_img_width = sheet_img_width//num_col # 130
    
    # a list to store each image in the sampleSheet_img
    sample_img_list = []
    # split the sampleSheet img into batch_size (here 16) images:
    tmp_count = 1
    for i in range(num_row):
        for j in range(num_col):
            start_row_pos = i*single_img_height
            end_row_pos = (i+1)*single_img_height
            start_col_pos = j*single_img_width
            end_col_pos = (j+1)*single_img_width
            
            single_sample_img = sampleSheet_img[start_row_pos:end_row_pos,start_col_pos:end_col_pos,:]
            
            if tmp_count <= batch_size:
                sample_img_list.append(single_sample_img)
            tmp_count += 1
    
    return sample_img_list


def generateTrainSet(len_featVec, dim):
    all_origin_img_vecs = [] # this is our feature space
    all_origin_img_names = []
    
    # newly modified: different from that in FLOWER_128_sub1000 and FLOWER_128:
    images_arr = np.load(srcRootDir_imgNpz)
    #images_arr.files
    images_list = list(images_arr['imgs'][:,0])
    
    # newly modified: different from that in FLOWER_128_sub1000 and FLOWER_128:
    """
    for (dirpath, dirnames, filenames) in os.walk(srcRootDir_originDataImg):
        for filename in filenames:
            if ".jpg" in filename:
    """
    for filename in images_list:
        #print("------------------deal with---------------------")
        #print(filename)
        #origin_img = cv2.imread(srcRootDir_originDataImg+filename)
        origin_img = cv2.imread(filename)
        origin_img_centCrop = my_center_crop(origin_img, min(origin_img.shape[0],origin_img.shape[1]))
        # resize using linear interpolation:
        origin_img_centCrop_resize = cv2.resize(origin_img_centCrop, dim)
        # also convert it to feature vector:
        origin_img_centCrop_resize_vec = image_to_feature_vector(origin_img_centCrop_resize)
        assert(len(origin_img_centCrop_resize_vec)==len_featVec)
        all_origin_img_vecs.append(origin_img_centCrop_resize_vec)
        all_origin_img_names.append(filename)
    
    return (np.array(all_origin_img_vecs), all_origin_img_names)


def my_center_crop(origin_img, crop_size):
    y,x,_ = origin_img.shape
    startx = x//2-(crop_size//2)
    starty = y//2-(crop_size//2)    
    origin_img_centCrop = origin_img[starty:starty+crop_size,startx:startx+crop_size]
    
    return origin_img_centCrop


def image_to_feature_vector(image):
    # Note: the image is already resized to a fixed size.
	# flatten the image into a list of raw pixel intensities:
    
	return image.flatten()


def combine_matchingResult(match_img_list):
    # combine the match_img together into a corresponding sheet
    
    (single_img_height, single_img_width, ch) = match_img_list[0].shape
    match_img_sheet = np.zeros((single_img_height*num_row,single_img_width*num_col,ch),dtype=np.uint8)
    
    for i in range(num_row):
        for j in range(num_col):
            start_row_pos = i*single_img_height
            end_row_pos = (i+1)*single_img_height
            start_col_pos = j*single_img_width
            end_col_pos = (j+1)*single_img_width
            match_img_idx = i*num_col + j
            
            if match_img_idx < batch_size:
                match_img_sheet[start_row_pos:end_row_pos,start_col_pos:end_col_pos,:] = match_img_list[match_img_idx]
    
    # save this sheet
    cv2.imwrite(dstImgName_NNmatchSheet, match_img_sheet)
    
    return


def query_NN_wrapper(sample_img_list):
    # this is a wrapper func!
    
    # first, get the training set from original images:
    len_featVec = len(image_to_feature_vector(sample_img_list[0]))
    dim = (sample_img_list[0].shape[1],sample_img_list[0].shape[0])
    trainSet_feats, all_origin_img_names = generateTrainSet(len_featVec, dim)
    
    neigh = NearestNeighbors(n_neighbors=1) # radius=0.4
    neigh.fit(trainSet_feats)
    
    # then, query:
    match_img_list = []
    match_distance_strs = ''
    for i in range(len(sample_img_list)):
        single_sample_img = sample_img_list[i]
        # get the query vector:
        single_sample_img_vec = image_to_feature_vector(single_sample_img)
        # NN to search:
        match_distance, match_idx = neigh.kneighbors([single_sample_img_vec], 1, return_distance=True)
        match_distance = match_distance[0][0]
        match_idx = match_idx[0][0]
        
        match_imgName = all_origin_img_names[match_idx].split('/')[-1]
        match_img = trainSet_feats[match_idx,:].reshape((dim[1],dim[0],3))
        match_img_list.append(match_img)
        # save the matching result:
        im_h = cv2.hconcat([single_sample_img, match_img])
        cv2.imwrite(dstRootDir_NNmatchResult+str(i+1)+'_'+match_imgName, im_h)
        # newly added: also save the corresponding match_distance into txt file:
        match_distance_strs += str(i+1)+'_'+match_imgName + ': match_distance = ' + str(match_distance) + '\n'
        
    # also combine the match_img together into a corresponding sheet!
    combine_matchingResult(match_img_list)
    # newly added: also save the corresponding match_distance into txt file:
    f = open(dstTxtName_matchDist, 'w')
    f.write(match_distance_strs)
    f.close()
    
    return





if __name__ == '__main__':
    # first, deal with the sample sheet:
    sample_img_list = dealWith_sampleSheet()
    #"""
    # for debug: save the generated sample images to visualize:
    for i in range(len(sample_img_list)):
        single_sample_img = sample_img_list[i]
        cv2.imwrite(dstRootDir_viewSampleSheetImgs+str(i+1)+'.jpg', single_sample_img)
    #"""
    
    # finally, query each single_sample_img into original dataset (FLOWER_128_xxx here);
    # also, save the matching results:
    query_NN_wrapper(sample_img_list)



