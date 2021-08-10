#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  5 23:47:57 2021

@author: guo.1648
"""

# final version.
# referenced from NN_getDist_testCode_forBiggan.py and
# NN_getRepThreshPairImg_testCode_forBiggan.py.

# This code does the following:
# (1) generate a 32x32 sample sheet from images in dir .../chenqi_random_samples/
# (2) do NN query as in NN_getDist_testCode_forBiggan.py
# (3) threshold the matched pairs as in NN_getRepThreshPairImg_testCode_forBiggan.py

# Note: for MNIST (grayscale) dataset, still we use 3 channels (where each channel is the same)
# to compute the L2 norm! <-- so that we don't need to change thresholds


import cv2
import os
import re
import numpy as np
from shutil import copyfile
from sklearn.neighbors import NearestNeighbors


NNmatchDist_threshold_values = [10000, 9000, 8000, 7000]
dstFolder_thresh_list = ['NNmatchResult_threshold10000/','NNmatchResult_threshold9000/','NNmatchResult_threshold8000/','NNmatchResult_threshold7000/']


"""
#### for FLOWER_128_sub1000: 1000 images dataset
srcDir_sampleSheetImgs = '/scratch/BigGAN-PyTorch/samples/BigGAN_FLOWER_128_sub1000_BigGANdeep_seed0_Gch128_Dch128_Gd2_Dd2_bs16_nDa64_nGa64_Glr1.0e-04_Dlr4.0e-04_Gnlinplace_relu_Dnlinplace_relu_Ginitortho_Dinitortho_Gattn64_Dattn64_Gshared_hier_ema/chenqi_random_samples/'
#srcRootDir_originDataImg = '' # NOT USED
srcRootDir_imgNpz = '/scratch/BigGAN-PyTorch/FLOWER_128_sub1000_imgs.npz'

dstRootDir = '/eecf/cbcsl/data100b/Chenqi/gan_results_for_presentation/biggan/NN_query/FLOWER_128_sub1000/Itr38950/'

# parameters:
im_size = 128
batch_size = 16 # i.e., each sample sheet is of 4x4 !!!!:
num_row = 4
num_col = 4
"""
"""
#### for FLOWER_128_sub2000: 2000 images dataset
srcDir_sampleSheetImgs = '/eecf/cbcsl/data100b/Chenqi/BigGAN-PyTorch/samples/BigGAN_FLOWER_128_sub2000_BigGANdeep_seed0_Gch128_Dch128_Gd2_Dd2_bs32_nDa64_nGa64_Glr1.0e-04_Dlr4.0e-04_Gnlinplace_relu_Dnlinplace_relu_Ginitortho_Dinitortho_Gattn64_Dattn64_Gshared_hier_ema/chenqi_random_samples/'
#srcRootDir_originDataImg = '' # NOT USED
srcRootDir_imgNpz = '/eecf/cbcsl/data100b/Chenqi/BigGAN-PyTorch/FLOWER_128_sub2000_imgs.npz'

dstRootDir = '/eecf/cbcsl/data100b/Chenqi/gan_results_for_presentation/biggan/NN_query/FLOWER_128_sub2000/Itr29700/'

# parameters:
im_size = 128
batch_size = 32 # i.e., the sample sheet is of 5x6+2 !!!!:
num_row = 7
num_col = 5
"""
"""
#### for FLOWER_128_sub4000: 4000 images dataset
srcDir_sampleSheetImgs = '/eecf/cbcsl/data100b/Chenqi/BigGAN-PyTorch/samples/BigGAN_FLOWER_128_sub4000_BigGANdeep_seed0_Gch128_Dch128_Gd2_Dd2_bs56_nDa64_nGa64_Glr1.0e-04_Dlr4.0e-04_Gnlinplace_relu_Dnlinplace_relu_Ginitortho_Dinitortho_Gattn64_Dattn64_Gshared_hier_ema/chenqi_random_samples/'
#srcRootDir_originDataImg = '' # NOT USED
srcRootDir_imgNpz = '/eecf/cbcsl/data100b/Chenqi/BigGAN-PyTorch/FLOWER_128_sub4000_imgs.npz'

dstRootDir = '/eecf/cbcsl/data100b/Chenqi/gan_results_for_presentation/biggan/NN_query/FLOWER_128_sub4000/Itr10700/'

# parameters:
im_size = 128
batch_size = 56 # i.e., each sample sheet is of 8x7 !!!!:
num_row = 8
num_col = 7
"""
"""
#### for FLOWER_128_sub6000: 6000 images dataset
srcDir_sampleSheetImgs = '/eecf/cbcsl/data100b/Chenqi/BigGAN-PyTorch/samples/BigGAN_FLOWER_128_sub6000_BigGANdeep_seed0_Gch128_Dch128_Gd2_Dd2_bs24_nDa64_nGa64_Glr1.0e-04_Dlr4.0e-04_Gnlinplace_relu_Dnlinplace_relu_Ginitortho_Dinitortho_Gattn64_Dattn64_Gshared_hier_ema/chenqi_random_samples/'
#srcRootDir_originDataImg = '' # NOT USED
srcRootDir_imgNpz = '/eecf/cbcsl/data100b/Chenqi/BigGAN-PyTorch/FLOWER_128_sub6000_imgs.npz'

dstRootDir = '/eecf/cbcsl/data100b/Chenqi/gan_results_for_presentation/biggan/NN_query/FLOWER_128_sub6000/Itr17300/'

# parameters:
im_size = 128
batch_size = 24 # i.e., each sample sheet is of 6x4 !!!!:
num_row = 6
num_col = 4
"""
"""
#### for FLOWER_128: 8189 images dataset (the original FLOWER dataset)
srcDir_sampleSheetImgs = '/scratch/BigGAN-PyTorch/samples/BigGAN_FLOWER_128_BigGANdeep_seed0_Gch128_Dch128_Gd2_Dd2_bs48_nDa64_nGa64_Glr1.0e-04_Dlr4.0e-04_Gnlinplace_relu_Dnlinplace_relu_Ginitortho_Dinitortho_Gattn64_Dattn64_Gshared_hier_ema/chenqi_random_samples/'
#srcRootDir_originDataImg = '' # NOT USED
srcRootDir_imgNpz = '/scratch/BigGAN-PyTorch/FLOWER_128_imgs.npz'

dstRootDir = '/eecf/cbcsl/data100b/Chenqi/gan_results_for_presentation/biggan/NN_query/FLOWER_128/Itr21500/'

# parameters:
im_size = 128
batch_size = 48 # i.e., each sample sheet is of 8x6 !!!!:
num_row = 8
num_col = 6
"""
"""
#### for CelebA_128_sub200: 200 images dataset
srcDir_sampleSheetImgs = '/eecf/cbcsl/data100b/Chenqi/BigGAN-PyTorch/samples/BigGAN_CelebA_128_sub200_BigGANdeep_seed0_Gch128_Dch128_Gd2_Dd2_bs32_nDa64_nGa64_Glr1.0e-04_Dlr4.0e-04_Gnlinplace_relu_Dnlinplace_relu_Ginitortho_Dinitortho_Gattn64_Dattn64_Gshared_hier_ema/chenqi_random_samples/'
#srcRootDir_originDataImg = '' # NOT USED
srcRootDir_imgNpz = '/eecf/cbcsl/data100b/Chenqi/BigGAN-PyTorch/CelebA_128_sub200_imgs.npz'

dstRootDir = '/eecf/cbcsl/data100b/Chenqi/gan_results_for_presentation/biggan/NN_query/CelebA_128_sub200/Itr17850/'

# parameters:
im_size = 128
batch_size = 32 # i.e., the sample sheet is of 5x6+2 !!!!:
num_row = 7
num_col = 5
"""
"""
#### for CelebA_128_sub600: 600 images dataset
srcDir_sampleSheetImgs = '/eecf/cbcsl/data100b/Chenqi/BigGAN-PyTorch/samples/BigGAN_CelebA_128_sub600_BigGANdeep_seed0_Gch128_Dch128_Gd2_Dd2_bs32_nDa64_nGa64_Glr1.0e-04_Dlr4.0e-04_Gnlinplace_relu_Dnlinplace_relu_Ginitortho_Dinitortho_Gattn64_Dattn64_Gshared_hier_ema/chenqi_random_samples/'
#srcRootDir_originDataImg = '' # NOT USED
srcRootDir_imgNpz = '/eecf/cbcsl/data100b/Chenqi/BigGAN-PyTorch/CelebA_128_sub600_imgs.npz'

dstRootDir = '/eecf/cbcsl/data100b/Chenqi/gan_results_for_presentation/biggan/NN_query/CelebA_128_sub600/Itr20450/'

# parameters:
im_size = 128
batch_size = 32 # i.e., the sample sheet is of 5x6+2 !!!!:
num_row = 7
num_col = 5
"""
"""
#### for CelebA_128_sub1000: 1000 images dataset
srcDir_sampleSheetImgs = '/scratch/BigGAN-PyTorch/samples/BigGAN_CelebA_128_sub1000_BigGANdeep_seed0_Gch128_Dch128_Gd2_Dd2_bs16_nDa64_nGa64_Glr1.0e-04_Dlr4.0e-04_Gnlinplace_relu_Dnlinplace_relu_Ginitortho_Dinitortho_Gattn64_Dattn64_Gshared_hier_ema/chenqi_random_samples/'
#srcRootDir_originDataImg = '' # NOT USED
srcRootDir_imgNpz = '/scratch/BigGAN-PyTorch/CelebA_128_sub1000_imgs.npz'

dstRootDir = '/eecf/cbcsl/data100b/Chenqi/gan_results_for_presentation/biggan/NN_query/CelebA_128_sub1000/Itr37400/'

# parameters:
im_size = 128
batch_size = 16 # i.e., each sample sheet is of 4x4 !!!!:
num_row = 4
num_col = 4
"""
"""
#### for CelebA_128_sub4000: 4000 images dataset
srcDir_sampleSheetImgs = '/eecf/cbcsl/data100b/Chenqi/BigGAN-PyTorch/samples/BigGAN_CelebA_128_sub4000_BigGANdeep_seed0_Gch128_Dch128_Gd2_Dd2_bs32_nDa64_nGa64_Glr1.0e-04_Dlr4.0e-04_Gnlinplace_relu_Dnlinplace_relu_Ginitortho_Dinitortho_Gattn64_Dattn64_Gshared_hier_ema/chenqi_random_samples/'
#srcRootDir_originDataImg = '' # NOT USED
srcRootDir_imgNpz = '/eecf/cbcsl/data100b/Chenqi/BigGAN-PyTorch/CelebA_128_sub4000_imgs.npz'

dstRootDir = '/eecf/cbcsl/data100b/Chenqi/gan_results_for_presentation/biggan/NN_query/CelebA_128_sub4000/Itr19600/'

# parameters:
im_size = 128
batch_size = 32 # i.e., the sample sheet is of 5x6+2 !!!!:
num_row = 7
num_col = 5
"""
"""
#### for CelebA_128_sub8000: 8000 images dataset
srcDir_sampleSheetImgs = '/eecf/cbcsl/data100b/Chenqi/BigGAN-PyTorch/samples/BigGAN_CelebA_128_sub8000_BigGANdeep_seed0_Gch128_Dch128_Gd2_Dd2_bs32_nDa64_nGa64_Glr1.0e-04_Dlr4.0e-04_Gnlinplace_relu_Dnlinplace_relu_Ginitortho_Dinitortho_Gattn64_Dattn64_Gshared_hier_ema/chenqi_random_samples/'
#srcRootDir_originDataImg = '' # NOT USED
srcRootDir_imgNpz = '/eecf/cbcsl/data100b/Chenqi/BigGAN-PyTorch/CelebA_128_sub8000_imgs.npz'

dstRootDir = '/eecf/cbcsl/data100b/Chenqi/gan_results_for_presentation/biggan/NN_query/CelebA_128_sub8000/Itr23550/'

# parameters:
im_size = 128
batch_size = 32 # i.e., the sample sheet is of 5x6+2 !!!!:
num_row = 7
num_col = 5
"""
"""
#### for MNIST_128_sub10000: 10000 images dataset
srcDir_sampleSheetImgs = '/eecf/cbcsl/data100b/Chenqi/BigGAN-PyTorch/samples/BigGAN_MNIST_128_sub10000_BigGANdeep_seed0_Gch128_Dch128_Gd2_Dd2_bs16_nDa64_nGa64_Glr1.0e-04_Dlr4.0e-04_Gnlinplace_relu_Dnlinplace_relu_Ginitortho_Dinitortho_Gattn64_Dattn64_Gshared_hier_ema/chenqi_random_samples/'
#srcRootDir_originDataImg = '' # NOT USED
srcRootDir_imgNpz = '/eecf/cbcsl/data100b/Chenqi/BigGAN-PyTorch/MNIST_128_sub10000_imgs.npz'

dstRootDir = '/eecf/cbcsl/data100b/Chenqi/gan_results_for_presentation/biggan/NN_query/MNIST_128_sub10000/Itr35600/'

# parameters:
im_size = 128
batch_size = 16 # i.e., each sample sheet is of 4x4 !!!!:
num_row = 4
num_col = 4
"""
"""
#### for MNIST_128_sub30000: 30000 images dataset
srcDir_sampleSheetImgs = '/eecf/cbcsl/data100b/Chenqi/BigGAN-PyTorch/samples/BigGAN_MNIST_128_sub30000_BigGANdeep_seed0_Gch128_Dch128_Gd2_Dd2_bs16_nDa64_nGa64_Glr1.0e-04_Dlr4.0e-04_Gnlinplace_relu_Dnlinplace_relu_Ginitortho_Dinitortho_Gattn64_Dattn64_Gshared_hier_ema/chenqi_random_samples/'
#srcRootDir_originDataImg = '' # NOT USED
srcRootDir_imgNpz = '/eecf/cbcsl/data100b/Chenqi/BigGAN-PyTorch/MNIST_128_sub30000_imgs.npz'

dstRootDir = '/eecf/cbcsl/data100b/Chenqi/gan_results_for_presentation/biggan/NN_query/MNIST_128_sub30000/Itr37300/'

# parameters:
im_size = 128
batch_size = 16 # i.e., each sample sheet is of 4x4 !!!!:
num_row = 4
num_col = 4
"""
"""
#### for MNIST_128_train: 60000 images dataset
srcDir_sampleSheetImgs = '/eecf/cbcsl/data100b/Chenqi/BigGAN-PyTorch/samples/BigGAN_MNIST_128_train_BigGANdeep_seed0_Gch128_Dch128_Gd2_Dd2_bs16_nDa64_nGa64_Glr1.0e-04_Dlr4.0e-04_Gnlinplace_relu_Dnlinplace_relu_Ginitortho_Dinitortho_Gattn64_Dattn64_Gshared_hier_ema/chenqi_random_samples/'
#srcRootDir_originDataImg = '' # NOT USED
srcRootDir_imgNpz = '/eecf/cbcsl/data100b/Chenqi/BigGAN-PyTorch/MNIST_128_train_imgs.npz'

dstRootDir = '/eecf/cbcsl/data100b/Chenqi/gan_results_for_presentation/biggan/NN_query/MNIST_128_train/Itr35850/'

# parameters:
im_size = 128
batch_size = 16 # i.e., each sample sheet is of 4x4 !!!!:
num_row = 4
num_col = 4
"""
"""
#### for LSUN_128_sub200: 200 images dataset
srcDir_sampleSheetImgs = '/eecf/cbcsl/data100b/Chenqi/BigGAN-PyTorch/samples/BigGAN_LSUN_128_sub200_BigGANdeep_seed0_Gch128_Dch128_Gd2_Dd2_bs48_nDa64_nGa64_Glr1.0e-04_Dlr4.0e-04_Gnlinplace_relu_Dnlinplace_relu_Ginitortho_Dinitortho_Gattn64_Dattn64_Gshared_hier_ema/chenqi_random_samples/'
#srcRootDir_originDataImg = '' # NOT USED
srcRootDir_imgNpz = '/eecf/cbcsl/data100b/Chenqi/BigGAN-PyTorch/LSUN_128_sub200_imgs.npz'

dstRootDir = '/eecf/cbcsl/data100b/Chenqi/gan_results_for_presentation/biggan/NN_query/LSUN_128_sub200/Itr12000/'

# parameters:
im_size = 128
batch_size = 48 # i.e., each sample sheet is of 8x6 !!!!:
num_row = 8
num_col = 6
"""
"""
#### for LSUN_128_sub1000: 1000 images dataset
srcDir_sampleSheetImgs = '/eecf/cbcsl/data100b/Chenqi/BigGAN-PyTorch/samples/BigGAN_LSUN_128_sub1000_BigGANdeep_seed0_Gch128_Dch128_Gd2_Dd2_bs48_nDa64_nGa64_Glr1.0e-04_Dlr4.0e-04_Gnlinplace_relu_Dnlinplace_relu_Ginitortho_Dinitortho_Gattn64_Dattn64_Gshared_hier_ema/chenqi_random_samples/'
#srcRootDir_originDataImg = '' # NOT USED
srcRootDir_imgNpz = '/eecf/cbcsl/data100b/Chenqi/BigGAN-PyTorch/LSUN_128_sub1000_imgs.npz'

dstRootDir = '/eecf/cbcsl/data100b/Chenqi/gan_results_for_presentation/biggan/NN_query/LSUN_128_sub1000/Itr13450/'

# parameters:
im_size = 128
batch_size = 48 # i.e., each sample sheet is of 8x6 !!!!:
num_row = 8
num_col = 6
"""
"""
#### for LSUN_128_sub5000: 5000 images dataset
srcDir_sampleSheetImgs = '/eecf/cbcsl/data100b/Chenqi/BigGAN-PyTorch/samples/BigGAN_LSUN_128_sub5000_BigGANdeep_seed0_Gch128_Dch128_Gd2_Dd2_bs48_nDa64_nGa64_Glr1.0e-04_Dlr4.0e-04_Gnlinplace_relu_Dnlinplace_relu_Ginitortho_Dinitortho_Gattn64_Dattn64_Gshared_hier_ema/chenqi_random_samples/'
#srcRootDir_originDataImg = '' # NOT USED
srcRootDir_imgNpz = '/eecf/cbcsl/data100b/Chenqi/BigGAN-PyTorch/LSUN_128_sub5000_imgs.npz'

dstRootDir = '/eecf/cbcsl/data100b/Chenqi/gan_results_for_presentation/biggan/NN_query/LSUN_128_sub5000/Itr9650/'

# parameters:
im_size = 128
batch_size = 48 # i.e., each sample sheet is of 8x6 !!!!:
num_row = 8
num_col = 6
"""
"""
#### for LSUN_128_sub10000: 10000 images dataset
srcDir_sampleSheetImgs = '/eecf/cbcsl/data100b/Chenqi/BigGAN-PyTorch/samples/BigGAN_LSUN_128_sub10000_BigGANdeep_seed0_Gch128_Dch128_Gd2_Dd2_bs48_nDa64_nGa64_Glr1.0e-04_Dlr4.0e-04_Gnlinplace_relu_Dnlinplace_relu_Ginitortho_Dinitortho_Gattn64_Dattn64_Gshared_hier_ema/chenqi_random_samples/'
#srcRootDir_originDataImg = '' # NOT USED
srcRootDir_imgNpz = '/eecf/cbcsl/data100b/Chenqi/BigGAN-PyTorch/LSUN_128_sub10000_imgs.npz'

dstRootDir = '/eecf/cbcsl/data100b/Chenqi/gan_results_for_presentation/biggan/NN_query/LSUN_128_sub10000/Itr12000/'

# parameters:
im_size = 128
batch_size = 48 # i.e., each sample sheet is of 8x6 !!!!:
num_row = 8
num_col = 6
"""
#"""
#### for LSUN_128_sub30000: 30000 images dataset
srcDir_sampleSheetImgs = '/eecf/cbcsl/data100b/Chenqi/BigGAN-PyTorch/samples/BigGAN_LSUN_128_sub30000_BigGANdeep_seed0_Gch128_Dch128_Gd2_Dd2_bs48_nDa64_nGa64_Glr1.0e-04_Dlr4.0e-04_Gnlinplace_relu_Dnlinplace_relu_Ginitortho_Dinitortho_Gattn64_Dattn64_Gshared_hier_ema/chenqi_random_samples/'
#srcRootDir_originDataImg = '' # NOT USED
srcRootDir_imgNpz = '/eecf/cbcsl/data100b/Chenqi/BigGAN-PyTorch/LSUN_128_sub30000_imgs.npz'

dstRootDir = '/eecf/cbcsl/data100b/Chenqi/gan_results_for_presentation/biggan/NN_query/LSUN_128_sub30000/Itr10400/'

# parameters:
im_size = 128
batch_size = 48 # i.e., each sample sheet is of 8x6 !!!!:
num_row = 8
num_col = 6
#"""





# for (1) and (2):
dstRootDir_viewSampleSheetImgs = dstRootDir + 'view_sampleSheetImgs/'
dstRootDir_NNmatchResult = dstRootDir + 'NNmatchResult/'
dstImgName_sampleSheetAll = dstRootDir + 'fakes.png'
dstImgName_NNmatchSheet = dstRootDir + 'NNmatchResultSheet.png'
dstTxtName_matchDist = dstRootDir + 'NNmatchDist.txt'
# for (3):
dstTxtName_matchDistThresh = dstRootDir + 'NNmatchDist_smallerThanThresh.txt'



def dealWith_sampleSheets():
    # the list to store each image in all the sampleSheet_imgs
    sample_img_list = []
    
    for (dirpath, dirnames, filenames) in os.walk(srcDir_sampleSheetImgs):
        #print(filenames)
        for filename in filenames:
            #print(filename)
            if ".jpg" in filename:
                print("------------------deal with---------------------")
                print(filename)
                fullImgName = srcDir_sampleSheetImgs + filename
                sampleSheet_img = cv2.imread(fullImgName)
                (sheet_img_height, sheet_img_width, ch) = sampleSheet_img.shape
                
                single_img_height = sheet_img_height//num_row # 130
                single_img_width = sheet_img_width//num_col # 130
                
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
    
    if len(sample_img_list) > 1024:
        sample_img_list = sample_img_list[:1024] # only keep 1025 imgs
    
    return sample_img_list


def generateSave_sampleSheetAll(sample_img_list):
    # generate and save the 32x32 sample sheet from sample_img_list
    
    (single_img_height, single_img_width, ch) = sample_img_list[0].shape
    sample_sheet_all = np.zeros((single_img_height*32,single_img_width*32,ch),dtype=np.uint8)
    
    for i in range(32):
        for j in range(32):
            start_row_pos = i*single_img_height
            end_row_pos = (i+1)*single_img_height
            start_col_pos = j*single_img_width
            end_col_pos = (j+1)*single_img_width
            match_img_idx = i*num_col + j
            
            if match_img_idx < 1024:
                sample_sheet_all[start_row_pos:end_row_pos,start_col_pos:end_col_pos,:] = sample_img_list[match_img_idx]
    
    # save this sheet
    cv2.imwrite(dstImgName_sampleSheetAll, sample_sheet_all)
    
    return


def image_to_feature_vector(image):
    # Note: the image is already resized to a fixed size.
	# flatten the image into a list of raw pixel intensities:
    
	return image.flatten()


def my_center_crop(origin_img, crop_size):
    y,x,_ = origin_img.shape
    startx = x//2-(crop_size//2)
    starty = y//2-(crop_size//2)    
    origin_img_centCrop = origin_img[starty:starty+crop_size,startx:startx+crop_size]
    
    return origin_img_centCrop


def generateTrainSet(len_featVec, dim):
    all_origin_img_vecs = [] # this is our feature space
    all_origin_img_names = []
    
    # newly modified: different from that in FLOWER_128_sub1000 and FLOWER_128:
    images_arr = np.load(srcRootDir_imgNpz)
    #images_arr.files
    images_list = list(images_arr['imgs'][:,0])
    
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


def combine_matchingResult(match_img_list):
    # combine the match_img together into a corresponding sheet
    
    (single_img_height, single_img_width, ch) = match_img_list[0].shape
    match_img_sheet = np.zeros((single_img_height*32,single_img_width*32,ch),dtype=np.uint8)
    
    for i in range(32):
        for j in range(32):
            start_row_pos = i*single_img_height
            end_row_pos = (i+1)*single_img_height
            start_col_pos = j*single_img_width
            end_col_pos = (j+1)*single_img_width
            match_img_idx = i*num_col + j
            
            if match_img_idx < 1024:
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


def threshNNpairs():
    match_distance_thresh_strs = ''
    
    for i in range(3):
        NNmatchDist_threshold_value = NNmatchDist_threshold_values[i]
        dstFolder_thresh = dstRootDir + dstFolder_thresh_list[i]
        
        match_distance_thresh_strs += '*******For threshold=' + str(NNmatchDist_threshold_value) + ':\n'
        
        match_num = 0
        tmp_match_distance_thresh_strs = ''
        for line in open(dstTxtName_matchDist):
            pairImgName = line.split(': match_distance = ')[0]
            l2Dist = float(line.split(': match_distance = ')[-1])
            #print()
            if l2Dist <= NNmatchDist_threshold_value:
                srcImgName = dstRootDir_NNmatchResult + pairImgName
                dstImgName = dstFolder_thresh + pairImgName
                copyfile(srcImgName, dstImgName)
                
                match_num += 1
                tmp_match_distance_thresh_strs += line
        
        match_percent = match_num / 1024 * 100
        match_distance_thresh_strs += 'match_num = ' + str(match_num) + '\n' \
                                     + 'total sample_num = ' + str(1024) + '\n' \
                                     + 'match_percent = ' + str(match_percent) + '%\n' \
                                     + tmp_match_distance_thresh_strs + '\n'
    
    # newly added: save the values related to match_distance smaller than thresh into txt file:
    f = open(dstTxtName_matchDistThresh, 'w')
    f.write(match_distance_thresh_strs)
    f.close()

    
    
    
    
    return
    

    
if __name__ == '__main__':
    #"""
    # (1) first, deal with all the sample sheets:
    sample_img_list = dealWith_sampleSheets()
    # (1) then, generate and save the 32x32 sample sheet from sample_img_list:
    generateSave_sampleSheetAll(sample_img_list)
    
    # for debug: save the generated sample images to visualize:
    for i in range(len(sample_img_list)):
        single_sample_img = sample_img_list[i]
        cv2.imwrite(dstRootDir_viewSampleSheetImgs+str(i+1)+'.jpg', single_sample_img)
    #"""
    #"""
    # (2) NN query each single_sample_img into original dataset;
    # also, save the matching results:
    query_NN_wrapper(sample_img_list)
    #"""
    
    # (3) threshold the matched pairs:
    threshNNpairs()
    
