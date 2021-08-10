#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 16:09:15 2020

@author: guo.1648
"""

# referenced from NN_query_testCode_forStylegan2.py,
# and NN_getDist_testCode_forBiggan.py.

# this code is for stylegan2 sample sheet.

# this code also do 1 NN matching for the generated images and the original images,
# but the purpose is to compute the corresponding matching distance for each result,
# and then use these matching distances and human perception for each mathing pair,
# to find out the NN distance threshold which is the largest matching distance satisfying
# 100% perceptual replication.

# Note: we will combine this stylegan2 result and the biggan result to find out the threshold!

import cv2
import os
import numpy as np
#from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestNeighbors
#from skimage import img_as_ubyte
#import torchvision.transforms as transforms

"""
#### for FLOWER_128: 8189 images dataset (the original FLOWER dataset)
src_sampleSheetImg = '/scratch/stylegan2/results/results_FLOWER_128/00000-stylegan2-FLOWER_128-1gpu-config-f/fakes002526.png'
srcRootDir_originDataImg = '/eecf/cbcsl/data100b/Chenqi/stylegan2/datasets_images/FLOWER_128/' # these images are generated from tfrecords using code mycode_loadImgFromTFrecords.py

dstRootDir_viewSampleSheetImgs = '/eecf/cbcsl/data100b/Chenqi/stylegan2/imgs/NN_query/FLOWER_128/fakes002526/view_sampleSheetImgs/'
dstRootDir_NNmatchResult = '/eecf/cbcsl/data100b/Chenqi/stylegan2/imgs/NN_query/FLOWER_128/fakes002526/NNmatchResult/'
dstImgName_NNmatchSheet = '/eecf/cbcsl/data100b/Chenqi/stylegan2/imgs/NN_query/FLOWER_128/fakes002526/NNmatchResultSheet.png'
dstTxtName_matchDist = '/eecf/cbcsl/data100b/Chenqi/stylegan2/imgs/NN_query/FLOWER_128/fakes002526/NNmatchDist.txt'
"""
"""
# for rebuttal:
#### for FLOWER_128: 8189 images dataset (the original FLOWER dataset)
src_sampleSheetImg = '/eecf/cbcsl/data100b/Chenqi/stylegan2/imgs/NN_query/FLOWER_128/fakes001925/fakes001925.png'
srcRootDir_originDataImg = '/eecf/cbcsl/data100b/Chenqi/stylegan2/datasets_images/FLOWER_128/jpg/' # these images are generated from tfrecords using code mycode_loadImgFromTFrecords.py

dstRootDir_viewSampleSheetImgs = '/eecf/cbcsl/data100b/Chenqi/stylegan2/imgs/NN_query/FLOWER_128/fakes001925/view_sampleSheetImgs/'
dstRootDir_NNmatchResult = '/eecf/cbcsl/data100b/Chenqi/stylegan2/imgs/NN_query/FLOWER_128/fakes001925/NNmatchResult/'
dstImgName_NNmatchSheet = '/eecf/cbcsl/data100b/Chenqi/stylegan2/imgs/NN_query/FLOWER_128/fakes001925/NNmatchResultSheet.png'
dstTxtName_matchDist = '/eecf/cbcsl/data100b/Chenqi/stylegan2/imgs/NN_query/FLOWER_128/fakes001925/NNmatchDist.txt'
"""

"""
#### for FLOWER_128_sub1000: 1000 images dataset (resume)
src_sampleSheetImg = '/eecf/cbcsl/data100b/Chenqi/stylegan2/results/results_FLOWER_128_sub1000_resume/00000-stylegan2-FLOWER_128_sub1000-1gpu-config-f/fakes003248.png'
srcRootDir_originDataImg = '/eecf/cbcsl/data100b/Chenqi/stylegan2/datasets_images/FLOWER_128_sub1000/' # these images are generated from tfrecords using code mycode_loadImgFromTFrecords.py

dstRootDir_viewSampleSheetImgs = '/eecf/cbcsl/data100b/Chenqi/stylegan2/imgs/NN_query/FLOWER_128_sub1000_resume/fakes003248/view_sampleSheetImgs/'
dstRootDir_NNmatchResult = '/eecf/cbcsl/data100b/Chenqi/stylegan2/imgs/NN_query/FLOWER_128_sub1000_resume/fakes003248/NNmatchResult/'
dstImgName_NNmatchSheet = '/eecf/cbcsl/data100b/Chenqi/stylegan2/imgs/NN_query/FLOWER_128_sub1000_resume/fakes003248/NNmatchResultSheet.png'
dstTxtName_matchDist = '/eecf/cbcsl/data100b/Chenqi/stylegan2/imgs/NN_query/FLOWER_128_sub1000_resume/fakes003248/NNmatchDist.txt'
"""
"""
# for rebuttal:
#### for FLOWER_128_sub1000: 1000 images dataset
src_sampleSheetImg = '/eecf/cbcsl/data100b/Chenqi/stylegan2/results/results_FLOWER_128_sub1000/00000-stylegan2-FLOWER_128_sub1000-1gpu-config-f/fakes001684.png'
srcRootDir_originDataImg = '/eecf/cbcsl/data100b/Chenqi/stylegan2/datasets_images/FLOWER_128_sub1000/jpg/' # these images are generated from tfrecords using code mycode_loadImgFromTFrecords.py

dstRootDir_viewSampleSheetImgs = '/eecf/cbcsl/data100b/Chenqi/stylegan2/imgs/NN_query/FLOWER_128_sub1000/fakes001684/view_sampleSheetImgs/'
dstRootDir_NNmatchResult = '/eecf/cbcsl/data100b/Chenqi/stylegan2/imgs/NN_query/FLOWER_128_sub1000/fakes001684/NNmatchResult/'
dstImgName_NNmatchSheet = '/eecf/cbcsl/data100b/Chenqi/stylegan2/imgs/NN_query/FLOWER_128_sub1000/fakes001684/NNmatchResultSheet.png'
dstTxtName_matchDist = '/eecf/cbcsl/data100b/Chenqi/stylegan2/imgs/NN_query/FLOWER_128_sub1000/fakes001684/NNmatchDist.txt'
"""

"""
#### for FLOWER_128_sub4000: 4000 images dataset (resume)
src_sampleSheetImg = '/eecf/cbcsl/data100b/Chenqi/stylegan2/results/results_FLOWER_128_sub4000_resume/00000-stylegan2-FLOWER_128_sub4000-1gpu-config-f/fakes003248.png'
srcRootDir_originDataImg = '/eecf/cbcsl/data100b/Chenqi/stylegan2/datasets_images/FLOWER_128_sub4000/' # these images are generated from tfrecords using code mycode_loadImgFromTFrecords.py

dstRootDir_viewSampleSheetImgs = '/eecf/cbcsl/data100b/Chenqi/stylegan2/imgs/NN_query/FLOWER_128_sub4000_resume/fakes003248/view_sampleSheetImgs/'
dstRootDir_NNmatchResult = '/eecf/cbcsl/data100b/Chenqi/stylegan2/imgs/NN_query/FLOWER_128_sub4000_resume/fakes003248/NNmatchResult/'
dstImgName_NNmatchSheet = '/eecf/cbcsl/data100b/Chenqi/stylegan2/imgs/NN_query/FLOWER_128_sub4000_resume/fakes003248/NNmatchResultSheet.png'
dstTxtName_matchDist = '/eecf/cbcsl/data100b/Chenqi/stylegan2/imgs/NN_query/FLOWER_128_sub4000_resume/fakes003248/NNmatchDist.txt'
"""
"""
# for rebuttal:
#### for FLOWER_128_sub4000: 4000 images dataset
src_sampleSheetImg = '/eecf/cbcsl/data100b/Chenqi/stylegan2/results/results_FLOWER_128_sub4000/00000-stylegan2-FLOWER_128_sub4000-1gpu-config-f/fakes001925.png'
srcRootDir_originDataImg = '/eecf/cbcsl/data100b/Chenqi/stylegan2/datasets_images/FLOWER_128_sub4000/jpg/' # these images are generated from tfrecords using code mycode_loadImgFromTFrecords.py

dstRootDir_viewSampleSheetImgs = '/eecf/cbcsl/data100b/Chenqi/stylegan2/imgs/NN_query/FLOWER_128_sub4000/fakes001925/view_sampleSheetImgs/'
dstRootDir_NNmatchResult = '/eecf/cbcsl/data100b/Chenqi/stylegan2/imgs/NN_query/FLOWER_128_sub4000/fakes001925/NNmatchResult/'
dstImgName_NNmatchSheet = '/eecf/cbcsl/data100b/Chenqi/stylegan2/imgs/NN_query/FLOWER_128_sub4000/fakes001925/NNmatchResultSheet.png'
dstTxtName_matchDist = '/eecf/cbcsl/data100b/Chenqi/stylegan2/imgs/NN_query/FLOWER_128_sub4000/fakes001925/NNmatchDist.txt'
"""

"""
#### for CelebA_128_sub200: 200 images dataset
src_sampleSheetImg = '/eecf/cbcsl/data100b/Chenqi/stylegan2/results/results_CelebA_128_sub200/00000-stylegan2-CelebA_128_sub200-1gpu-config-f/fakes007700.png'
srcRootDir_originDataImg = '/eecf/cbcsl/data100b/Chenqi/stylegan2/datasets_images/CelebA_128_sub200/jpg/' # these images are generated from tfrecords using code mycode_loadImgFromTFrecords.py

dstRootDir_viewSampleSheetImgs = '/eecf/cbcsl/data100b/Chenqi/stylegan2/imgs/NN_query/CelebA_128_sub200/fakes007700/view_sampleSheetImgs/'
dstRootDir_NNmatchResult = '/eecf/cbcsl/data100b/Chenqi/stylegan2/imgs/NN_query/CelebA_128_sub200/fakes007700/NNmatchResult/'
dstImgName_NNmatchSheet = '/eecf/cbcsl/data100b/Chenqi/stylegan2/imgs/NN_query/CelebA_128_sub200/fakes007700/NNmatchResultSheet.png'
dstTxtName_matchDist = '/eecf/cbcsl/data100b/Chenqi/stylegan2/imgs/NN_query/CelebA_128_sub200/fakes007700/NNmatchDist.txt'
"""
"""
#### for CelebA_128_sub600: 600 images dataset
src_sampleSheetImg = '/eecf/cbcsl/data100b/Chenqi/stylegan2/results/results_CelebA_128_sub600/00000-stylegan2-CelebA_128_sub600-1gpu-config-f/fakes005414.png'
srcRootDir_originDataImg = '/eecf/cbcsl/data100b/Chenqi/stylegan2/datasets_images/CelebA_128_sub600/jpg/' # these images are generated from tfrecords using code mycode_loadImgFromTFrecords.py

dstRootDir_viewSampleSheetImgs = '/eecf/cbcsl/data100b/Chenqi/stylegan2/imgs/NN_query/CelebA_128_sub600/fakes005414/view_sampleSheetImgs/'
dstRootDir_NNmatchResult = '/eecf/cbcsl/data100b/Chenqi/stylegan2/imgs/NN_query/CelebA_128_sub600/fakes005414/NNmatchResult/'
dstImgName_NNmatchSheet = '/eecf/cbcsl/data100b/Chenqi/stylegan2/imgs/NN_query/CelebA_128_sub600/fakes005414/NNmatchResultSheet.png'
dstTxtName_matchDist = '/eecf/cbcsl/data100b/Chenqi/stylegan2/imgs/NN_query/CelebA_128_sub600/fakes005414/NNmatchDist.txt'
"""
"""
#### for CelebA_128_sub1000: 1000 images dataset
src_sampleSheetImg = '/eecf/cbcsl/data100b/Chenqi/stylegan2/results/results_CelebA_128_sub1000/00000-stylegan2-CelebA_128_sub1000-1gpu-config-f/fakes004933.png'
srcRootDir_originDataImg = '/eecf/cbcsl/data100b/Chenqi/stylegan2/datasets_images/CelebA_128_sub1000/jpg/' # these images are generated from tfrecords using code mycode_loadImgFromTFrecords.py

dstRootDir_viewSampleSheetImgs = '/eecf/cbcsl/data100b/Chenqi/stylegan2/imgs/NN_query/CelebA_128_sub1000/fakes004933/view_sampleSheetImgs/'
dstRootDir_NNmatchResult = '/eecf/cbcsl/data100b/Chenqi/stylegan2/imgs/NN_query/CelebA_128_sub1000/fakes004933/NNmatchResult/'
dstImgName_NNmatchSheet = '/eecf/cbcsl/data100b/Chenqi/stylegan2/imgs/NN_query/CelebA_128_sub1000/fakes004933/NNmatchResultSheet.png'
dstTxtName_matchDist = '/eecf/cbcsl/data100b/Chenqi/stylegan2/imgs/NN_query/CelebA_128_sub1000/fakes004933/NNmatchDist.txt'
"""
"""
#### for CelebA_128_sub4000: 4000 images dataset
src_sampleSheetImg = '/eecf/cbcsl/data100b/Chenqi/stylegan2/results/results_CelebA_128_sub4000/00000-stylegan2-CelebA_128_sub4000-1gpu-config-f/fakes003369.png'
srcRootDir_originDataImg = '/eecf/cbcsl/data100b/Chenqi/stylegan2/datasets_images/CelebA_128_sub4000/' # these images are generated from tfrecords using code mycode_loadImgFromTFrecords.py

dstRootDir_viewSampleSheetImgs = '/eecf/cbcsl/data100b/Chenqi/stylegan2/imgs/NN_query/CelebA_128_sub4000/fakes003369/view_sampleSheetImgs/'
dstRootDir_NNmatchResult = '/eecf/cbcsl/data100b/Chenqi/stylegan2/imgs/NN_query/CelebA_128_sub4000/fakes003369/NNmatchResult/'
dstImgName_NNmatchSheet = '/eecf/cbcsl/data100b/Chenqi/stylegan2/imgs/NN_query/CelebA_128_sub4000/fakes003369/NNmatchResultSheet.png'
dstTxtName_matchDist = '/eecf/cbcsl/data100b/Chenqi/stylegan2/imgs/NN_query/CelebA_128_sub4000/fakes003369/NNmatchDist.txt'
"""
"""
#### for CelebA_128_sub8000: 8000 images dataset
src_sampleSheetImg = '/eecf/cbcsl/data100b/Chenqi/stylegan2/results/results_CelebA_128_sub8000/00000-stylegan2-CelebA_128_sub8000-1gpu-config-f/fakes001684.png'
srcRootDir_originDataImg = '/eecf/cbcsl/data100b/Chenqi/stylegan2/datasets_images/CelebA_128_sub8000/' # these images are generated from tfrecords using code mycode_loadImgFromTFrecords.py

dstRootDir_viewSampleSheetImgs = '/eecf/cbcsl/data100b/Chenqi/stylegan2/imgs/NN_query/CelebA_128_sub8000/fakes001684/view_sampleSheetImgs/'
dstRootDir_NNmatchResult = '/eecf/cbcsl/data100b/Chenqi/stylegan2/imgs/NN_query/CelebA_128_sub8000/fakes001684/NNmatchResult/'
dstImgName_NNmatchSheet = '/eecf/cbcsl/data100b/Chenqi/stylegan2/imgs/NN_query/CelebA_128_sub8000/fakes001684/NNmatchResultSheet.png'
dstTxtName_matchDist = '/eecf/cbcsl/data100b/Chenqi/stylegan2/imgs/NN_query/CelebA_128_sub8000/fakes001684/NNmatchDist.txt'
"""
"""
#### for MNIST_128_sub10000: 10000 images dataset
src_sampleSheetImg = '/eecf/cbcsl/data100b/Chenqi/stylegan2/results/results_MNIST_128_sub10000/00002-stylegan2-MNIST_128_sub10000-1gpu-config-f/fakes005173.png'
srcRootDir_originDataImg = '/eecf/cbcsl/data100b/Chenqi/stylegan2/datasets_images/MNIST_128_sub10000/' # these images are generated from tfrecords using code mycode_loadImgFromTFrecords.py

dstRootDir_viewSampleSheetImgs = '/eecf/cbcsl/data100b/Chenqi/stylegan2/imgs/NN_query/MNIST_128_sub10000/fakes005173/view_sampleSheetImgs/'
dstRootDir_NNmatchResult = '/eecf/cbcsl/data100b/Chenqi/stylegan2/imgs/NN_query/MNIST_128_sub10000/fakes005173/NNmatchResult/'
dstImgName_NNmatchSheet = '/eecf/cbcsl/data100b/Chenqi/stylegan2/imgs/NN_query/MNIST_128_sub10000/fakes005173/NNmatchResultSheet.png'
dstTxtName_matchDist = '/eecf/cbcsl/data100b/Chenqi/stylegan2/imgs/NN_query/MNIST_128_sub10000/fakes005173/NNmatchDist.txt'
"""
"""
#### for MNIST_128_sub10000: 10000 images dataset, bi:
src_sampleSheetImg = '/eecf/cbcsl/data100b/Chenqi/stylegan2/results/results_MNIST_128_sub10000/00002-stylegan2-MNIST_128_sub10000-1gpu-config-f/fakes005173.png'
srcRootDir_originDataImg = '/eecf/cbcsl/data100b/Chenqi/stylegan2/datasets_images/MNIST_128_sub10000/' # these images are generated from tfrecords using code mycode_loadImgFromTFrecords.py

dstRootDir_viewSampleSheetImgs = '/eecf/cbcsl/data100b/Chenqi/stylegan2/imgs/NN_query/MNIST_128_sub10000_3ch/fakes005173/view_sampleSheetImgs/'
dstRootDir_NNmatchResult = '/eecf/cbcsl/data100b/Chenqi/stylegan2/imgs/NN_query/MNIST_128_sub10000_3ch/fakes005173/NNmatchResult/'
dstImgName_NNmatchSheet = '/eecf/cbcsl/data100b/Chenqi/stylegan2/imgs/NN_query/MNIST_128_sub10000_3ch/fakes005173/NNmatchResultSheet.png'
dstTxtName_matchDist = '/eecf/cbcsl/data100b/Chenqi/stylegan2/imgs/NN_query/MNIST_128_sub10000_3ch/fakes005173/NNmatchDist.txt'
"""
"""
#### for MNIST_128_sub30000: 30000 images dataset
src_sampleSheetImg = '/eecf/cbcsl/data100b/Chenqi/stylegan2/results/results_MNIST_128_sub30000/00000-stylegan2-MNIST_128_sub30000-1gpu-config-f/fakes005053.png'
srcRootDir_originDataImg = '/eecf/cbcsl/data100b/Chenqi/stylegan2/datasets_images/MNIST_128_sub30000/' # these images are generated from tfrecords using code mycode_loadImgFromTFrecords.py

dstRootDir_viewSampleSheetImgs = '/eecf/cbcsl/data100b/Chenqi/stylegan2/imgs/NN_query/MNIST_128_sub30000/fakes005053/view_sampleSheetImgs/'
dstRootDir_NNmatchResult = '/eecf/cbcsl/data100b/Chenqi/stylegan2/imgs/NN_query/MNIST_128_sub30000/fakes005053/NNmatchResult/'
dstImgName_NNmatchSheet = '/eecf/cbcsl/data100b/Chenqi/stylegan2/imgs/NN_query/MNIST_128_sub30000/fakes005053/NNmatchResultSheet.png'
dstTxtName_matchDist = '/eecf/cbcsl/data100b/Chenqi/stylegan2/imgs/NN_query/MNIST_128_sub30000/fakes005053/NNmatchDist.txt'
"""
"""
#### for MNIST_128_sub30000: 30000 images dataset, bi:
src_sampleSheetImg = '/eecf/cbcsl/data100b/Chenqi/stylegan2/results/results_MNIST_128_sub30000/00000-stylegan2-MNIST_128_sub30000-1gpu-config-f/fakes005053.png'
srcRootDir_originDataImg = '/eecf/cbcsl/data100b/Chenqi/stylegan2/datasets_images/MNIST_128_sub30000_bi/' # these images are generated from tfrecords using code mycode_loadImgFromTFrecords.py

dstRootDir_viewSampleSheetImgs = '/eecf/cbcsl/data100b/Chenqi/stylegan2/imgs/NN_query/MNIST_128_sub30000_bi/fakes005053/view_sampleSheetImgs/'
dstRootDir_NNmatchResult = '/eecf/cbcsl/data100b/Chenqi/stylegan2/imgs/NN_query/MNIST_128_sub30000_bi/fakes005053/NNmatchResult/'
dstImgName_NNmatchSheet = '/eecf/cbcsl/data100b/Chenqi/stylegan2/imgs/NN_query/MNIST_128_sub30000_bi/fakes005053/NNmatchResultSheet.png'
dstTxtName_matchDist = '/eecf/cbcsl/data100b/Chenqi/stylegan2/imgs/NN_query/MNIST_128_sub30000_bi/fakes005053/NNmatchDist.txt'
"""
"""
#### for MNIST_128_train: 60000 images dataset
src_sampleSheetImg = '/eecf/cbcsl/data100b/Chenqi/stylegan2/results/results_MNIST_128_train/00000-stylegan2-MNIST_128_train-1gpu-config-f/fakes003609.png'
srcRootDir_originDataImg = '/eecf/cbcsl/data100b/Chenqi/data/MNIST/resized/train/train_60000/' # these images are just the whole MNIST resized training set

dstRootDir_viewSampleSheetImgs = '/eecf/cbcsl/data100b/Chenqi/stylegan2/imgs/NN_query/MNIST_128_train/fakes003609/view_sampleSheetImgs/'
dstRootDir_NNmatchResult = '/eecf/cbcsl/data100b/Chenqi/stylegan2/imgs/NN_query/MNIST_128_train/fakes003609/NNmatchResult/'
dstImgName_NNmatchSheet = '/eecf/cbcsl/data100b/Chenqi/stylegan2/imgs/NN_query/MNIST_128_train/fakes003609/NNmatchResultSheet.png'
dstTxtName_matchDist = '/eecf/cbcsl/data100b/Chenqi/stylegan2/imgs/NN_query/MNIST_128_train/fakes003609/NNmatchDist.txt'
"""
"""
#### for LSUN_128_sub10000: 10000 images dataset
src_sampleSheetImg = '/eecf/cbcsl/data100b/Chenqi/stylegan2/results/results_LSUN_128_sub10000/00000-stylegan2-LSUN_128_sub10000-1gpu-config-f/fakes004812.png'
srcRootDir_originDataImg = '/eecf/cbcsl/data100b/Chenqi/stylegan2/datasets_images/LSUN_128_sub10000/'

dstRootDir_viewSampleSheetImgs = '/eecf/cbcsl/data100b/Chenqi/stylegan2/imgs/NN_query/LSUN_128_sub10000/fakes004812/view_sampleSheetImgs/'
dstRootDir_NNmatchResult = '/eecf/cbcsl/data100b/Chenqi/stylegan2/imgs/NN_query/LSUN_128_sub10000/fakes004812/NNmatchResult/'
dstImgName_NNmatchSheet = '/eecf/cbcsl/data100b/Chenqi/stylegan2/imgs/NN_query/LSUN_128_sub10000/fakes004812/NNmatchResultSheet.png'
dstTxtName_matchDist = '/eecf/cbcsl/data100b/Chenqi/stylegan2/imgs/NN_query/LSUN_128_sub10000/fakes004812/NNmatchDist.txt'
"""
"""
#### for LSUN_128_sub30000: 30000 images dataset
src_sampleSheetImg = '/eecf/cbcsl/data100b/Chenqi/stylegan2/results/results_LSUN_128_sub30000/00020-stylegan2-LSUN_128_sub30000-1gpu-config-f/fakes004692.png'
srcRootDir_originDataImg = '/eecf/cbcsl/data100b/Chenqi/stylegan2/datasets_images/LSUN_128_sub30000/'

dstRootDir_viewSampleSheetImgs = '/eecf/cbcsl/data100b/Chenqi/stylegan2/imgs/NN_query/LSUN_128_sub30000/fakes004692/view_sampleSheetImgs/'
dstRootDir_NNmatchResult = '/eecf/cbcsl/data100b/Chenqi/stylegan2/imgs/NN_query/LSUN_128_sub30000/fakes004692/NNmatchResult/'
dstImgName_NNmatchSheet = '/eecf/cbcsl/data100b/Chenqi/stylegan2/imgs/NN_query/LSUN_128_sub30000/fakes004692/NNmatchResultSheet.png'
dstTxtName_matchDist = '/eecf/cbcsl/data100b/Chenqi/stylegan2/imgs/NN_query/LSUN_128_sub30000/fakes004692/NNmatchDist.txt'
"""
"""
#### for LSUN_128_sub60000: 60000 images dataset
src_sampleSheetImg = '/eecf/cbcsl/data100b/Chenqi/stylegan2/results/results_LSUN_128_sub60000/00000-stylegan2-LSUN_128_sub60000-1gpu-config-f/fakes006497.png'
srcRootDir_originDataImg = '/eecf/cbcsl/data100b/Chenqi/stylegan2/datasets_images/LSUN_128_sub60000/'

dstRootDir_viewSampleSheetImgs = '/eecf/cbcsl/data100b/Chenqi/stylegan2/imgs/NN_query/LSUN_128_sub60000/fakes006497/view_sampleSheetImgs/'
dstRootDir_NNmatchResult = '/eecf/cbcsl/data100b/Chenqi/stylegan2/imgs/NN_query/LSUN_128_sub60000/fakes006497/NNmatchResult/'
dstImgName_NNmatchSheet = '/eecf/cbcsl/data100b/Chenqi/stylegan2/imgs/NN_query/LSUN_128_sub60000/fakes006497/NNmatchResultSheet.png'
dstTxtName_matchDist = '/eecf/cbcsl/data100b/Chenqi/stylegan2/imgs/NN_query/LSUN_128_sub60000/fakes006497/NNmatchDist.txt'
"""
"""
#### for LSUN_128_sub1000_resume: 1000 images dataset
src_sampleSheetImg = '/eecf/cbcsl/data100b/Chenqi/stylegan2/results/results_LSUN_128_sub1000_resume/00000-stylegan2-LSUN_128_sub1000-1gpu-config-f/fakes002165.png'
srcRootDir_originDataImg = '/eecf/cbcsl/data100b/Chenqi/stylegan2/datasets_images/LSUN_128_sub1000/'

dstRootDir_viewSampleSheetImgs = '/eecf/cbcsl/data100b/Chenqi/stylegan2/imgs/NN_query/LSUN_128_sub1000_resume/fakes002165/view_sampleSheetImgs/'
dstRootDir_NNmatchResult = '/eecf/cbcsl/data100b/Chenqi/stylegan2/imgs/NN_query/LSUN_128_sub1000_resume/fakes002165/NNmatchResult/'
dstImgName_NNmatchSheet = '/eecf/cbcsl/data100b/Chenqi/stylegan2/imgs/NN_query/LSUN_128_sub1000_resume/fakes002165/NNmatchResultSheet.png'
dstTxtName_matchDist = '/eecf/cbcsl/data100b/Chenqi/stylegan2/imgs/NN_query/LSUN_128_sub1000_resume/fakes002165/NNmatchDist.txt'
"""
"""
#### for LSUN_128_sub5000_resume: 5000 images dataset
src_sampleSheetImg = '/eecf/cbcsl/data100b/Chenqi/stylegan2/results/results_LSUN_128_sub5000_resume/00000-stylegan2-LSUN_128_sub5000-1gpu-config-f/fakes000000.png'
srcRootDir_originDataImg = '/eecf/cbcsl/data100b/Chenqi/stylegan2/datasets_images/LSUN_128_sub5000/'

dstRootDir_viewSampleSheetImgs = '/eecf/cbcsl/data100b/Chenqi/stylegan2/imgs/NN_query/LSUN_128_sub5000_resume/fakes000000/view_sampleSheetImgs/'
dstRootDir_NNmatchResult = '/eecf/cbcsl/data100b/Chenqi/stylegan2/imgs/NN_query/LSUN_128_sub5000_resume/fakes000000/NNmatchResult/'
dstImgName_NNmatchSheet = '/eecf/cbcsl/data100b/Chenqi/stylegan2/imgs/NN_query/LSUN_128_sub5000_resume/fakes000000/NNmatchResultSheet.png'
dstTxtName_matchDist = '/eecf/cbcsl/data100b/Chenqi/stylegan2/imgs/NN_query/LSUN_128_sub5000_resume/fakes000000/NNmatchDist.txt'
"""
"""
#### for LSUN_128_sub200: 200 images dataset
src_sampleSheetImg = '/eecf/cbcsl/data100b/Chenqi/stylegan2/results/results_LSUN_128_sub200/00000-stylegan2-LSUN_128_sub200-1gpu-config-f/fakes006497.png'
srcRootDir_originDataImg = '/eecf/cbcsl/data100b/Chenqi/stylegan2/datasets_images/LSUN_128_sub200/'

dstRootDir_viewSampleSheetImgs = '/eecf/cbcsl/data100b/Chenqi/stylegan2/imgs/NN_query/LSUN_128_sub200/fakes006497/view_sampleSheetImgs/'
dstRootDir_NNmatchResult = '/eecf/cbcsl/data100b/Chenqi/stylegan2/imgs/NN_query/LSUN_128_sub200/fakes006497/NNmatchResult/'
dstImgName_NNmatchSheet = '/eecf/cbcsl/data100b/Chenqi/stylegan2/imgs/NN_query/LSUN_128_sub200/fakes006497/NNmatchResultSheet.png'
dstTxtName_matchDist = '/eecf/cbcsl/data100b/Chenqi/stylegan2/imgs/NN_query/LSUN_128_sub200/fakes006497/NNmatchDist.txt'
"""

"""
# parameters:
im_size = 128
# note: the sample sheet is of 32x32:
num_row = 32
num_col = 32
"""



### for rebuttal: CIFAR10:
"""
#### for CIFAR10_32_sub1000: 1000 images dataset
src_sampleSheetImg = '/eecf/cbcsl/data100b/Chenqi/stylegan2/results/results_CIFAR10_32_sub1000/00000-stylegan2-CIFAR10_32_sub1000-1gpu-config-f/fakes002813.png'
srcRootDir_originDataImg = '/eecf/cbcsl/data100b/Chenqi/stylegan2/datasets_images/CIFAR10_32_sub1000/'

dstRootDir_viewSampleSheetImgs = '/eecf/cbcsl/data100b/Chenqi/stylegan2/imgs/NN_query/CIFAR10_32_sub1000/fakes002813/view_sampleSheetImgs/'
dstRootDir_NNmatchResult = '/eecf/cbcsl/data100b/Chenqi/stylegan2/imgs/NN_query/CIFAR10_32_sub1000/fakes002813/NNmatchResult/'
dstImgName_NNmatchSheet = '/eecf/cbcsl/data100b/Chenqi/stylegan2/imgs/NN_query/CIFAR10_32_sub1000/fakes002813/NNmatchResultSheet.png'
dstTxtName_matchDist = '/eecf/cbcsl/data100b/Chenqi/stylegan2/imgs/NN_query/CIFAR10_32_sub1000/fakes002813/NNmatchDist.txt'
"""
"""
#### for CIFAR10_32_sub4000: 4000 images dataset
src_sampleSheetImg = '/eecf/cbcsl/data100b/Chenqi/stylegan2/results/results_CIFAR10_32_sub4000/00000-stylegan2-CIFAR10_32_sub4000-1gpu-config-f/fakes003014.png'
srcRootDir_originDataImg = '/eecf/cbcsl/data100b/Chenqi/stylegan2/datasets_images/CIFAR10_32_sub4000/'

dstRootDir_viewSampleSheetImgs = '/eecf/cbcsl/data100b/Chenqi/stylegan2/imgs/NN_query/CIFAR10_32_sub4000/fakes003014/view_sampleSheetImgs/'
dstRootDir_NNmatchResult = '/eecf/cbcsl/data100b/Chenqi/stylegan2/imgs/NN_query/CIFAR10_32_sub4000/fakes003014/NNmatchResult/'
dstImgName_NNmatchSheet = '/eecf/cbcsl/data100b/Chenqi/stylegan2/imgs/NN_query/CIFAR10_32_sub4000/fakes003014/NNmatchResultSheet.png'
dstTxtName_matchDist = '/eecf/cbcsl/data100b/Chenqi/stylegan2/imgs/NN_query/CIFAR10_32_sub4000/fakes003014/NNmatchDist.txt'
"""
"""
#### for CIFAR10_32_sub8000: 8000 images dataset
src_sampleSheetImg = '/eecf/cbcsl/data100b/Chenqi/stylegan2/results/results_CIFAR10_32_sub8000/00000-stylegan2-CIFAR10_32_sub8000-1gpu-config-f/fakes003014.png'
srcRootDir_originDataImg = '/eecf/cbcsl/data100b/Chenqi/stylegan2/datasets_images/CIFAR10_32_sub8000/'

dstRootDir_viewSampleSheetImgs = '/eecf/cbcsl/data100b/Chenqi/stylegan2/imgs/NN_query/CIFAR10_32_sub8000/fakes003014/view_sampleSheetImgs/'
dstRootDir_NNmatchResult = '/eecf/cbcsl/data100b/Chenqi/stylegan2/imgs/NN_query/CIFAR10_32_sub8000/fakes003014/NNmatchResult/'
dstImgName_NNmatchSheet = '/eecf/cbcsl/data100b/Chenqi/stylegan2/imgs/NN_query/CIFAR10_32_sub8000/fakes003014/NNmatchResultSheet.png'
dstTxtName_matchDist = '/eecf/cbcsl/data100b/Chenqi/stylegan2/imgs/NN_query/CIFAR10_32_sub8000/fakes003014/NNmatchDist.txt'
"""
"""
#### for CIFAR10_32_sub10000: 10000 images dataset
src_sampleSheetImg = '/eecf/cbcsl/data100b/Chenqi/stylegan2/results/results_CIFAR10_32_sub10000/00000-stylegan2-CIFAR10_32_sub10000-1gpu-config-f/fakes002009.png'
srcRootDir_originDataImg = '/eecf/cbcsl/data100b/Chenqi/stylegan2/datasets_images/CIFAR10_32_sub10000/'

dstRootDir_viewSampleSheetImgs = '/eecf/cbcsl/data100b/Chenqi/stylegan2/imgs/NN_query/CIFAR10_32_sub10000/fakes002009/view_sampleSheetImgs/'
dstRootDir_NNmatchResult = '/eecf/cbcsl/data100b/Chenqi/stylegan2/imgs/NN_query/CIFAR10_32_sub10000/fakes002009/NNmatchResult/'
dstImgName_NNmatchSheet = '/eecf/cbcsl/data100b/Chenqi/stylegan2/imgs/NN_query/CIFAR10_32_sub10000/fakes002009/NNmatchResultSheet.png'
dstTxtName_matchDist = '/eecf/cbcsl/data100b/Chenqi/stylegan2/imgs/NN_query/CIFAR10_32_sub10000/fakes002009/NNmatchDist.txt'
"""

"""
# parameters: for CIFAR10:
im_size = 32
# note: the sample sheet is of 32x32:
num_row = 32
num_col = 32
"""



### for rebuttal: image size 256x256:
"""
#### for FLOWER_256_sub1000: 1000 images dataset
src_sampleSheetImg = '/eecf/cbcsl/data100b/Chenqi/stylegan2/results/results_FLOWER_256_sub1000/00002-stylegan2-FLOWER_256_sub1000-1gpu-config-f/fakes004435.png'
srcRootDir_originDataImg = '/eecf/cbcsl/data100b/Chenqi/stylegan2/datasets_images/FLOWER_256_sub1000/'

dstRootDir_viewSampleSheetImgs = '/eecf/cbcsl/data100b/Chenqi/stylegan2/imgs/NN_query/FLOWER_256_sub1000/fakes004435/view_sampleSheetImgs/'
dstRootDir_NNmatchResult = '/eecf/cbcsl/data100b/Chenqi/stylegan2/imgs/NN_query/FLOWER_256_sub1000/fakes004435/NNmatchResult/'
dstImgName_NNmatchSheet = '/eecf/cbcsl/data100b/Chenqi/stylegan2/imgs/NN_query/FLOWER_256_sub1000/fakes004435/NNmatchResultSheet.png'
dstTxtName_matchDist = '/eecf/cbcsl/data100b/Chenqi/stylegan2/imgs/NN_query/FLOWER_256_sub1000/fakes004435/NNmatchDist.txt'
"""
"""
#### for FLOWER_256_sub4000: 4000 images dataset
src_sampleSheetImg = '/eecf/cbcsl/data100b/Chenqi/stylegan2/results/results_FLOWER_256_sub4000/00002-stylegan2-FLOWER_256_sub4000-1gpu-config-f/fakes006128.png'
srcRootDir_originDataImg = '/eecf/cbcsl/data100b/Chenqi/stylegan2/datasets_images/FLOWER_256_sub4000/'

dstRootDir_viewSampleSheetImgs = '/eecf/cbcsl/data100b/Chenqi/stylegan2/imgs/NN_query/FLOWER_256_sub4000/fakes006128/view_sampleSheetImgs/'
dstRootDir_NNmatchResult = '/eecf/cbcsl/data100b/Chenqi/stylegan2/imgs/NN_query/FLOWER_256_sub4000/fakes006128/NNmatchResult/'
dstImgName_NNmatchSheet = '/eecf/cbcsl/data100b/Chenqi/stylegan2/imgs/NN_query/FLOWER_256_sub4000/fakes006128/NNmatchResultSheet.png'
dstTxtName_matchDist = '/eecf/cbcsl/data100b/Chenqi/stylegan2/imgs/NN_query/FLOWER_256_sub4000/fakes006128/NNmatchDist.txt'
"""
"""
#### for FLOWER_256_sub6000: 6000 images dataset
src_sampleSheetImg = '/eecf/cbcsl/data100b/Chenqi/stylegan2/results/results_FLOWER_256_sub6000/00002-stylegan2-FLOWER_256_sub6000-1gpu-config-f/fakes006290.png'
srcRootDir_originDataImg = '/eecf/cbcsl/data100b/Chenqi/stylegan2/datasets_images/FLOWER_256_sub6000/'

dstRootDir_viewSampleSheetImgs = '/eecf/cbcsl/data100b/Chenqi/stylegan2/imgs/NN_query/FLOWER_256_sub6000/fakes006290/view_sampleSheetImgs/'
dstRootDir_NNmatchResult = '/eecf/cbcsl/data100b/Chenqi/stylegan2/imgs/NN_query/FLOWER_256_sub6000/fakes006290/NNmatchResult/'
dstImgName_NNmatchSheet = '/eecf/cbcsl/data100b/Chenqi/stylegan2/imgs/NN_query/FLOWER_256_sub6000/fakes006290/NNmatchResultSheet.png'
dstTxtName_matchDist = '/eecf/cbcsl/data100b/Chenqi/stylegan2/imgs/NN_query/FLOWER_256_sub6000/fakes006290/NNmatchDist.txt'
"""
#"""
#### for FLOWER_256: 8189 images dataset
src_sampleSheetImg = '/eecf/cbcsl/data100b/Chenqi/stylegan2/results/results_FLOWER_256/00002-stylegan2-FLOWER_256-1gpu-config-f/fakes006209.png'
srcRootDir_originDataImg = '/eecf/cbcsl/data100b/Chenqi/stylegan2/datasets_images/FLOWER_256/jpg/'

dstRootDir_viewSampleSheetImgs = '/eecf/cbcsl/data100b/Chenqi/stylegan2/imgs/NN_query/FLOWER_256/fakes006209/view_sampleSheetImgs/'
dstRootDir_NNmatchResult = '/eecf/cbcsl/data100b/Chenqi/stylegan2/imgs/NN_query/FLOWER_256/fakes006209/NNmatchResult/'
dstImgName_NNmatchSheet = '/eecf/cbcsl/data100b/Chenqi/stylegan2/imgs/NN_query/FLOWER_256/fakes006209/NNmatchResultSheet.png'
dstTxtName_matchDist = '/eecf/cbcsl/data100b/Chenqi/stylegan2/imgs/NN_query/FLOWER_256/fakes006209/NNmatchDist.txt'
#"""

#"""
# parameters: only for 256x256:
im_size = 256
# note: the sample sheet is of 32x32:
num_row = 16
num_col = 30
#"""



# Newly added: only used for MNIST dataset:
# binarize the images!
#biFlag = True # for MNIST dataset
biFlag = False # for other (RGB or grayscale) dataset


def dealWith_sampleSheet():
    sampleSheet_img = cv2.imread(src_sampleSheetImg)
    (sheet_img_height, sheet_img_width, ch) = sampleSheet_img.shape
    
    single_img_height = sheet_img_height//num_row # 128
    single_img_width = sheet_img_width//num_col # 128
    
    # a list to store each image in the sampleSheet_img
    sample_img_list = []
    # split the sampleSheet img into batch_size (here 16) images:
    for i in range(num_row):
        for j in range(num_col):
            start_row_pos = i*single_img_height
            end_row_pos = (i+1)*single_img_height
            start_col_pos = j*single_img_width
            end_col_pos = (j+1)*single_img_width
            
            single_sample_img = sampleSheet_img[start_row_pos:end_row_pos,start_col_pos:end_col_pos,:]
            
            # Newly added:
            if biFlag:
                single_sample_img_gray = single_sample_img[:,:,0]
                _,single_sample_img = cv2.threshold(single_sample_img_gray,127,255,cv2.THRESH_BINARY)
            
            sample_img_list.append(single_sample_img)
    
    return sample_img_list


def image_to_feature_vector(image):
    # Note: the image is already resized to a fixed size.
	# flatten the image into a list of raw pixel intensities:
    
	return image.flatten()


def generateTrainSet(len_featVec, dim):
    all_origin_img_vecs = [] # this is our feature space
    all_origin_img_names = []
    
    for (dirpath, dirnames, filenames) in os.walk(srcRootDir_originDataImg):
        for filename in filenames:
            if ".jpg" in filename or ".png" in filename:
                print("------------------deal with---------------------")
                print(filename)
                origin_img = cv2.imread(srcRootDir_originDataImg+filename)
                if biFlag:
                    origin_img = origin_img[:,:,0]
                """
                # NO need to do this here: already 128x128 !
                origin_img_centCrop = my_center_crop(origin_img, min(origin_img.shape[0],origin_img.shape[1]))
                # resize using linear interpolation:
                origin_img_centCrop_resize = cv2.resize(origin_img_centCrop, dim)
                """
                # also convert it to feature vector:
                origin_img_centCrop_resize_vec = image_to_feature_vector(origin_img)
                assert(len(origin_img_centCrop_resize_vec)==len_featVec)
                all_origin_img_vecs.append(origin_img_centCrop_resize_vec)
                all_origin_img_names.append(filename)
    
    return (np.array(all_origin_img_vecs), all_origin_img_names)


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
        
        match_imgName = all_origin_img_names[match_idx]
        if biFlag:
            match_img = trainSet_feats[match_idx,:].reshape((dim[1],dim[0],1))
        else:
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
        cv2.imwrite(dstRootDir_viewSampleSheetImgs+str(i+1)+'.png', single_sample_img)
    #"""
    
    # finally, query each single_sample_img into original dataset (FLOWER_128_xxx here);
    # also, save the matching results:
    query_NN_wrapper(sample_img_list)



