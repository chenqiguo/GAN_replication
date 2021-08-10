#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 26 21:10:57 2020

@author: guo.1648
"""

# put the NN matching pair images with L2-norm distance smaller than threshold = 10000
# into the dst folder.

import re
import numpy as np
from shutil import copyfile

NNmatchDist_threshold_value = 10000 #10000 #9000 #8000



"""
#### for FLOWER_128_sub1000: 1000 images dataset
# for samples38800:
srcRootDir_img = '/scratch/BigGAN-PyTorch/imgs/NN_query/FLOWER_128_sub1000/samples38800/NNmatchResult/'
srcTxtFile = '/scratch/BigGAN-PyTorch/imgs/NN_query/FLOWER_128_sub1000/samples38800/NNmatchDist.txt'

dstRootDir_img = '/eecf/cbcsl/data100b/Chenqi/gan_results_for_presentation/biggan/NN_query/FLOWER_128_sub1000/samples38800/NNmatchResult_threshold8000/'

total_sample_num = 16
dstTxtName_matchDistThresh = '/scratch/BigGAN-PyTorch/imgs/NN_query/FLOWER_128_sub1000/samples38800/NNmatchDist_smallerThanThresh.txt'
"""
"""
#### for FLOWER_128: 8189 images dataset (the original FLOWER dataset)
# for samples21400:
srcRootDir_img = '/scratch/BigGAN-PyTorch/imgs/NN_query/FLOWER_128/samples21400/NNmatchResult/'
srcTxtFile = '/scratch/BigGAN-PyTorch/imgs/NN_query/FLOWER_128/samples21400/NNmatchDist.txt'

dstRootDir_img = '/eecf/cbcsl/data100b/Chenqi/gan_results_for_presentation/biggan/NN_query/FLOWER_128/samples21400/NNmatchResult_threshold8000/'
total_sample_num = 48
dstTxtName_matchDistThresh = '/scratch/BigGAN-PyTorch/imgs/NN_query/FLOWER_128/samples21400/NNmatchDist_smallerThanThresh.txt'
"""
"""
#### for CelebA_128_sub1000: 1000 images dataset
# for samples27000:
srcRootDir_img = '/scratch/BigGAN-PyTorch/imgs/NN_query/CelebA_128_sub1000/samples27000/NNmatchResult/'
srcTxtFile = '/scratch/BigGAN-PyTorch/imgs/NN_query/CelebA_128_sub1000/samples27000/NNmatchDist.txt'

dstRootDir_img = '/eecf/cbcsl/data100b/Chenqi/gan_results_for_presentation/biggan/NN_query/CelebA_128_sub1000/samples27000/NNmatchResult_threshold10000/'
total_sample_num = 16
dstTxtName_matchDistThresh = '/scratch/BigGAN-PyTorch/imgs/NN_query/CelebA_128_sub1000/samples27000/NNmatchDist_smallerThanThresh.txt'
"""
"""
#### for CelebA_128_sub4000: 4000 images dataset
# for samples16200:
srcRootDir_img = '/eecf/cbcsl/data100b/Chenqi/BigGAN-PyTorch/imgs/NN_query/CelebA_128_sub4000/samples16200/NNmatchResult/'
srcTxtFile = '/eecf/cbcsl/data100b/Chenqi/BigGAN-PyTorch/imgs/NN_query/CelebA_128_sub4000/samples16200/NNmatchDist.txt'

dstRootDir_img = '/eecf/cbcsl/data100b/Chenqi/gan_results_for_presentation/biggan/NN_query/CelebA_128_sub4000/samples16200/NNmatchResult_threshold8000/'
total_sample_num = 32
dstTxtName_matchDistThresh = '/eecf/cbcsl/data100b/Chenqi/BigGAN-PyTorch/imgs/NN_query/CelebA_128_sub4000/samples16200/NNmatchDist_smallerThanThresh.txt'
"""
"""
#### for CelebA_128_sub8000: 8000 images dataset
# for samples20400:
srcRootDir_img = '/eecf/cbcsl/data100b/Chenqi/BigGAN-PyTorch/imgs/NN_query/CelebA_128_sub8000/samples20400/NNmatchResult/'
srcTxtFile = '/eecf/cbcsl/data100b/Chenqi/BigGAN-PyTorch/imgs/NN_query/CelebA_128_sub8000/samples20400/NNmatchDist.txt'

dstRootDir_img = '/eecf/cbcsl/data100b/Chenqi/gan_results_for_presentation/biggan/NN_query/CelebA_128_sub8000/samples20400/NNmatchResult_threshold10000/'
total_sample_num = 32
dstTxtName_matchDistThresh = '/eecf/cbcsl/data100b/Chenqi/BigGAN-PyTorch/imgs/NN_query/CelebA_128_sub8000/samples20400/NNmatchDist_smallerThanThresh.txt'
"""
"""
#### for CelebA_128_sub200: 200 images dataset
# for samples17800:
srcRootDir_img = '/eecf/cbcsl/data100b/Chenqi/BigGAN-PyTorch/imgs/NN_query/CelebA_128_sub200/samples17800/NNmatchResult/'
srcTxtFile = '/eecf/cbcsl/data100b/Chenqi/BigGAN-PyTorch/imgs/NN_query/CelebA_128_sub200/samples17800/NNmatchDist.txt'

dstRootDir_img = '/eecf/cbcsl/data100b/Chenqi/gan_results_for_presentation/biggan/NN_query/CelebA_128_sub200/samples17800/NNmatchResult_threshold8000/'
total_sample_num = 32
dstTxtName_matchDistThresh = '/eecf/cbcsl/data100b/Chenqi/BigGAN-PyTorch/imgs/NN_query/CelebA_128_sub200/samples17800/NNmatchDist_smallerThanThresh.txt'
"""
"""
#### for FLOWER_128_sub4000_bs56: 4000 images dataset
# for samples10000:
srcRootDir_img = '/eecf/cbcsl/data100b/Chenqi/BigGAN-PyTorch/imgs/NN_query/FLOWER_128_sub4000_bs56/samples10000/NNmatchResult/'
srcTxtFile = '/eecf/cbcsl/data100b/Chenqi/BigGAN-PyTorch/imgs/NN_query/FLOWER_128_sub4000_bs56/samples10000/NNmatchDist.txt'

dstRootDir_img = '/eecf/cbcsl/data100b/Chenqi/gan_results_for_presentation/biggan/NN_query/FLOWER_128_sub4000_bs56/samples10000/NNmatchResult_threshold10000/'
total_sample_num = 56
dstTxtName_matchDistThresh = '/eecf/cbcsl/data100b/Chenqi/BigGAN-PyTorch/imgs/NN_query/FLOWER_128_sub4000_bs56/samples10000/NNmatchDist_smallerThanThresh.txt'
"""
"""
#### for FLOWER_128_sub6000: 6000 images dataset
# for samples17000:
srcRootDir_img = '/eecf/cbcsl/data100b/Chenqi/BigGAN-PyTorch/imgs/NN_query/FLOWER_128_sub6000/samples17000/NNmatchResult/'
srcTxtFile = '/eecf/cbcsl/data100b/Chenqi/BigGAN-PyTorch/imgs/NN_query/FLOWER_128_sub6000/samples17000/NNmatchDist.txt'

dstRootDir_img = '/eecf/cbcsl/data100b/Chenqi/gan_results_for_presentation/biggan/NN_query/FLOWER_128_sub6000/samples17000/NNmatchResult_threshold8000/'
total_sample_num = 24
dstTxtName_matchDistThresh = '/eecf/cbcsl/data100b/Chenqi/BigGAN-PyTorch/imgs/NN_query/FLOWER_128_sub6000/samples17000/NNmatchDist_smallerThanThresh.txt'
"""
"""
#### for MNIST_128_train: 60000 images dataset
# for samples20600:
srcRootDir_img = '/eecf/cbcsl/data100b/Chenqi/BigGAN-PyTorch/imgs/NN_query/MNIST_128_train/samples34400/NNmatchResult/'
srcTxtFile = '/eecf/cbcsl/data100b/Chenqi/BigGAN-PyTorch/imgs/NN_query/MNIST_128_train/samples34400/NNmatchDist.txt'

dstRootDir_img = '/eecf/cbcsl/data100b/Chenqi/gan_results_for_presentation/biggan/NN_query/MNIST_128_train/samples34400/NNmatchResult_threshold10000/'
total_sample_num = 16
dstTxtName_matchDistThresh = '/eecf/cbcsl/data100b/Chenqi/BigGAN-PyTorch/imgs/NN_query/MNIST_128_train/samples34400/NNmatchDist_smallerThanThresh.txt'
"""
"""
#### for CelebA_128_sub600: 600 images dataset
# for samples17400:
srcRootDir_img = '/eecf/cbcsl/data100b/Chenqi/BigGAN-PyTorch/imgs/NN_query/CelebA_128_sub600/samples17400/NNmatchResult/'
srcTxtFile = '/eecf/cbcsl/data100b/Chenqi/BigGAN-PyTorch/imgs/NN_query/CelebA_128_sub600/samples17400/NNmatchDist.txt'

dstRootDir_img = '/eecf/cbcsl/data100b/Chenqi/gan_results_for_presentation/biggan/NN_query/CelebA_128_sub600/samples17400/NNmatchResult_threshold8000/'
total_sample_num = 32
dstTxtName_matchDistThresh = '/eecf/cbcsl/data100b/Chenqi/BigGAN-PyTorch/imgs/NN_query/CelebA_128_sub600/samples17400/NNmatchDist_smallerThanThresh.txt'
"""
"""
#### for FLOWER_128_sub2000: 2000 images dataset
# for samples24600:
srcRootDir_img = '/eecf/cbcsl/data100b/Chenqi/BigGAN-PyTorch/imgs/NN_query/FLOWER_128_sub2000/samples24600/NNmatchResult/'
srcTxtFile = '/eecf/cbcsl/data100b/Chenqi/BigGAN-PyTorch/imgs/NN_query/FLOWER_128_sub2000/samples24600/NNmatchDist.txt'

dstRootDir_img = '/eecf/cbcsl/data100b/Chenqi/gan_results_for_presentation/biggan/NN_query/FLOWER_128_sub2000/samples24600/NNmatchResult_threshold10000/'
total_sample_num = 32
dstTxtName_matchDistThresh = '/eecf/cbcsl/data100b/Chenqi/BigGAN-PyTorch/imgs/NN_query/FLOWER_128_sub2000/samples24600/NNmatchDist_smallerThanThresh.txt'
"""
"""
#### for MNIST_128_sub10000: 10000 images dataset
# for samples24000:
srcRootDir_img = '/eecf/cbcsl/data100b/Chenqi/BigGAN-PyTorch/imgs/NN_query/MNIST_128_sub10000/samples34250/NNmatchResult/'
srcTxtFile = '/eecf/cbcsl/data100b/Chenqi/BigGAN-PyTorch/imgs/NN_query/MNIST_128_sub10000/samples34250/NNmatchDist.txt'

dstRootDir_img = '/eecf/cbcsl/data100b/Chenqi/gan_results_for_presentation/biggan/NN_query/MNIST_128_sub10000/samples34250/NNmatchResult_threshold8000/'
total_sample_num = 16
dstTxtName_matchDistThresh = '/eecf/cbcsl/data100b/Chenqi/BigGAN-PyTorch/imgs/NN_query/MNIST_128_sub10000/samples34250/NNmatchDist_smallerThanThresh.txt'
"""
"""
#### for MNIST_128_sub30000: 30000 images dataset
# for samples32000:
srcRootDir_img = '/eecf/cbcsl/data100b/Chenqi/BigGAN-PyTorch/imgs/NN_query/MNIST_128_sub30000/samples32000/NNmatchResult/'
srcTxtFile = '/eecf/cbcsl/data100b/Chenqi/BigGAN-PyTorch/imgs/NN_query/MNIST_128_sub30000/samples32000/NNmatchDist.txt'

dstRootDir_img = '/eecf/cbcsl/data100b/Chenqi/gan_results_for_presentation/biggan/NN_query/MNIST_128_sub30000/samples32000/NNmatchResult_threshold8000/'
total_sample_num = 16
dstTxtName_matchDistThresh = '/eecf/cbcsl/data100b/Chenqi/BigGAN-PyTorch/imgs/NN_query/MNIST_128_sub30000/samples32000/NNmatchDist_smallerThanThresh.txt'
"""
"""
#### for LSUN_128_sub200: 200 images dataset
# for samples11800:
srcRootDir_img = '/eecf/cbcsl/data100b/Chenqi/BigGAN-PyTorch/imgs/NN_query/LSUN_128_sub200/samples11800/NNmatchResult/'
srcTxtFile = '/eecf/cbcsl/data100b/Chenqi/BigGAN-PyTorch/imgs/NN_query/LSUN_128_sub200/samples11800/NNmatchDist.txt'

dstRootDir_img = '/eecf/cbcsl/data100b/Chenqi/gan_results_for_presentation/biggan/NN_query/LSUN_128_sub200/samples11800/NNmatchResult_threshold10000/'
total_sample_num = 48
dstTxtName_matchDistThresh = '/eecf/cbcsl/data100b/Chenqi/gan_results_for_presentation/biggan/NN_query/LSUN_128_sub200/samples11800/NNmatchDist_smallerThanThresh.txt'
"""


# for rebuttal: incepv3 comb pix:
"""
NNmatchDist_threshold_value = 42
#### for FLOWER_128_sub1000: 1000 images dataset
# for Itr38950:
srcRootDir_img = '/eecf/cbcsl/data100b/Chenqi/gan_results_for_presentation/biggan/NNquery_inceptionv3_pixelwise/FLOWER_128_sub1000/Itr38950/NNmatchResult/'
srcTxtFile = '/eecf/cbcsl/data100b/Chenqi/gan_results_for_presentation/biggan/NNquery_inceptionv3_pixelwise/FLOWER_128_sub1000/Itr38950/NNmatchDist.txt'

dstRootDir_img = '/eecf/cbcsl/data100b/Chenqi/gan_results_for_presentation/biggan/NNquery_inceptionv3_pixelwise/FLOWER_128_sub1000/Itr38950/NNmatchResult_threshold42/'
total_sample_num = 1024
dstTxtName_matchDistThresh = '/eecf/cbcsl/data100b/Chenqi/gan_results_for_presentation/biggan/NNquery_inceptionv3_pixelwise/FLOWER_128_sub1000/Itr38950/NNmatchDist_smallerThanThresh.txt'
"""
"""
NNmatchDist_threshold_value = 38
#### for FLOWER_128_sub2000: 2000 images dataset
# for Itr29700:
srcRootDir_img = '/eecf/cbcsl/data100b/Chenqi/gan_results_for_presentation/biggan/NNquery_inceptionv3_pixelwise/FLOWER_128_sub2000/Itr29700/NNmatchResult/'
srcTxtFile = '/eecf/cbcsl/data100b/Chenqi/gan_results_for_presentation/biggan/NNquery_inceptionv3_pixelwise/FLOWER_128_sub2000/Itr29700/NNmatchDist.txt'

dstRootDir_img = '/eecf/cbcsl/data100b/Chenqi/gan_results_for_presentation/biggan/NNquery_inceptionv3_pixelwise/FLOWER_128_sub2000/Itr29700/NNmatchResult_threshold38/'
total_sample_num = 1024
dstTxtName_matchDistThresh = '/eecf/cbcsl/data100b/Chenqi/gan_results_for_presentation/biggan/NNquery_inceptionv3_pixelwise/FLOWER_128_sub2000/Itr29700/NNmatchDist_smallerThanThresh.txt'
"""
"""
NNmatchDist_threshold_value = 42
#### for FLOWER_128_sub4000: 4000 images dataset
# for Itr10700:
srcRootDir_img = '/eecf/cbcsl/data100b/Chenqi/gan_results_for_presentation/biggan/NNquery_inceptionv3_pixelwise/FLOWER_128_sub4000/Itr10700/NNmatchResult/'
srcTxtFile = '/eecf/cbcsl/data100b/Chenqi/gan_results_for_presentation/biggan/NNquery_inceptionv3_pixelwise/FLOWER_128_sub4000/Itr10700/NNmatchDist.txt'

dstRootDir_img = '/eecf/cbcsl/data100b/Chenqi/gan_results_for_presentation/biggan/NNquery_inceptionv3_pixelwise/FLOWER_128_sub4000/Itr10700/NNmatchResult_threshold42/'
total_sample_num = 1024
dstTxtName_matchDistThresh = '/eecf/cbcsl/data100b/Chenqi/gan_results_for_presentation/biggan/NNquery_inceptionv3_pixelwise/FLOWER_128_sub4000/Itr10700/NNmatchDist_smallerThanThresh.txt'
"""
"""
NNmatchDist_threshold_value = 42
#### for FLOWER_128_sub6000: 6000 images dataset
# for Itr17300:
srcRootDir_img = '/eecf/cbcsl/data100b/Chenqi/gan_results_for_presentation/biggan/NNquery_inceptionv3_pixelwise/FLOWER_128_sub6000/Itr17300/NNmatchResult/'
srcTxtFile = '/eecf/cbcsl/data100b/Chenqi/gan_results_for_presentation/biggan/NNquery_inceptionv3_pixelwise/FLOWER_128_sub6000/Itr17300/NNmatchDist.txt'

dstRootDir_img = '/eecf/cbcsl/data100b/Chenqi/gan_results_for_presentation/biggan/NNquery_inceptionv3_pixelwise/FLOWER_128_sub6000/Itr17300/NNmatchResult_threshold42/'
total_sample_num = 1024
dstTxtName_matchDistThresh = '/eecf/cbcsl/data100b/Chenqi/gan_results_for_presentation/biggan/NNquery_inceptionv3_pixelwise/FLOWER_128_sub6000/Itr17300/NNmatchDist_smallerThanThresh.txt'
"""
"""
NNmatchDist_threshold_value = 33
#### for FLOWER_128: 8189 images dataset
# for Itr21500:
srcRootDir_img = '/eecf/cbcsl/data100b/Chenqi/gan_results_for_presentation/biggan/NNquery_inceptionv3_pixelwise/FLOWER_128/Itr21500/NNmatchResult/'
srcTxtFile = '/eecf/cbcsl/data100b/Chenqi/gan_results_for_presentation/biggan/NNquery_inceptionv3_pixelwise/FLOWER_128/Itr21500/NNmatchDist.txt'

dstRootDir_img = '/eecf/cbcsl/data100b/Chenqi/gan_results_for_presentation/biggan/NNquery_inceptionv3_pixelwise/FLOWER_128/Itr21500/NNmatchResult_threshold33/'
total_sample_num = 1024
dstTxtName_matchDistThresh = '/eecf/cbcsl/data100b/Chenqi/gan_results_for_presentation/biggan/NNquery_inceptionv3_pixelwise/FLOWER_128/Itr21500/NNmatchDist_smallerThanThresh.txt'
"""
"""
NNmatchDist_threshold_value = 33
#### for FLOWER_128_sub1000: 1000 images dataset
# for fakes003248:
srcRootDir_img = '/eecf/cbcsl/data100b/Chenqi/gan_results_for_presentation/stylegan2/NNquery_inceptionv3_pixelwise/FLOWER_128_sub1000/fakes003248/NNmatchResult/'
srcTxtFile = '/eecf/cbcsl/data100b/Chenqi/gan_results_for_presentation/stylegan2/NNquery_inceptionv3_pixelwise/FLOWER_128_sub1000/fakes003248/NNmatchDist.txt'

dstRootDir_img = '/eecf/cbcsl/data100b/Chenqi/gan_results_for_presentation/stylegan2/NNquery_inceptionv3_pixelwise/FLOWER_128_sub1000/fakes003248/NNmatchResult_threshold33/'
total_sample_num = 1024
dstTxtName_matchDistThresh = '/eecf/cbcsl/data100b/Chenqi/gan_results_for_presentation/stylegan2/NNquery_inceptionv3_pixelwise/FLOWER_128_sub1000/fakes003248/NNmatchDist_smallerThanThresh.txt'
"""
"""
NNmatchDist_threshold_value = 39
#### for FLOWER_128_sub4000: 4000 images dataset
# for fakes003248:
srcRootDir_img = '/eecf/cbcsl/data100b/Chenqi/gan_results_for_presentation/stylegan2/NNquery_inceptionv3_pixelwise/FLOWER_128_sub4000/fakes003248/NNmatchResult/'
srcTxtFile = '/eecf/cbcsl/data100b/Chenqi/gan_results_for_presentation/stylegan2/NNquery_inceptionv3_pixelwise/FLOWER_128_sub4000/fakes003248/NNmatchDist.txt'

dstRootDir_img = '/eecf/cbcsl/data100b/Chenqi/gan_results_for_presentation/stylegan2/NNquery_inceptionv3_pixelwise/FLOWER_128_sub4000/fakes003248/NNmatchResult_threshold39/'
total_sample_num = 1024
dstTxtName_matchDistThresh = '/eecf/cbcsl/data100b/Chenqi/gan_results_for_presentation/stylegan2/NNquery_inceptionv3_pixelwise/FLOWER_128_sub4000/fakes003248/NNmatchDist_smallerThanThresh.txt'
"""

NNmatchDist_threshold_value = 42
#### for FLOWER_128: 8189 images dataset
# for fakes002526:
srcRootDir_img = '/eecf/cbcsl/data100b/Chenqi/gan_results_for_presentation/stylegan2/NNquery_inceptionv3_pixelwise/FLOWER_128/fakes002526/NNmatchResult/'
srcTxtFile = '/eecf/cbcsl/data100b/Chenqi/gan_results_for_presentation/stylegan2/NNquery_inceptionv3_pixelwise/FLOWER_128/fakes002526/NNmatchDist.txt'

dstRootDir_img = '/eecf/cbcsl/data100b/Chenqi/gan_results_for_presentation/stylegan2/NNquery_inceptionv3_pixelwise/FLOWER_128/fakes002526/NNmatchResult_threshold42/'
total_sample_num = 1024
dstTxtName_matchDistThresh = '/eecf/cbcsl/data100b/Chenqi/gan_results_for_presentation/stylegan2/NNquery_inceptionv3_pixelwise/FLOWER_128/fakes002526/NNmatchDist_smallerThanThresh.txt'




if __name__ == '__main__':
    match_distance_thresh_strs = ''
    match_num = 0
    for line in open(srcTxtFile):
        pairImgName = line.split(': match_distance = ')[0]
        l2Dist = float(line.split(': match_distance = ')[-1])
        #print()
        if l2Dist <= NNmatchDist_threshold_value:
            srcImgName = srcRootDir_img + pairImgName
            dstImgName = dstRootDir_img + pairImgName
            copyfile(srcImgName, dstImgName)
            
            match_num += 1
            match_distance_thresh_strs += line
        
    
    match_percent = match_num / total_sample_num * 100
    match_distance_thresh_strs = 'match_num = ' + str(match_num) + '\n' \
                                 + 'total sample_num = ' + str(total_sample_num) + '\n' \
                                 + 'match_percent = ' + str(match_percent) + '%\n' \
                                 + match_distance_thresh_strs
    # newly added: save the values related to match_distance smaller than thresh into txt file:
    f = open(dstTxtName_matchDistThresh, 'w')
    f.write(match_distance_thresh_strs)
    f.close()




