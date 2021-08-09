#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 12:35:52 2021

@author: guo.1648
"""

# allocate biggan & stylegan imgs for human behavioral experiment.


import cv2
import os
import re
import numpy as np
from shutil import copyfile


n_img = 100


bigganExp_list = ['CelebA_128_sub200/', 'CelebA_128_sub600/', 'CelebA_128_sub1000/', 'CelebA_128_sub4000/', 'CelebA_128_sub8000/',
                  'FLOWER_128_sub1000/', 'FLOWER_128_sub2000/', 'FLOWER_128_sub4000/', 'FLOWER_128_sub6000/', 'FLOWER_128/',
                  'LSUN_128_sub200/', 'LSUN_128_sub1000/', 'LSUN_128_sub5000/', 'LSUN_128_sub10000/',
                  'MNIST_128_sub10000/', 'MNIST_128_sub30000/', 'MNIST_128_train/']
styleganExp_list = ['CelebA_128_sub1000/', 'CelebA_128_sub4000/', 'CelebA_128_sub8000/',
                   'FLOWER_128_sub1000_resume/', 'FLOWER_128_sub4000_resume/', 'FLOWER_128/',
                   'LSUN_128_sub200/', 'LSUN_128_sub1000_resume/', 'LSUN_128_sub5000_resume/', 'LSUN_128_sub10000/', 'LSUN_128_sub30000/',
                   'MNIST_128_sub10000/', 'MNIST_128_sub30000/', 'MNIST_128_train/']


dstRootDir = '/eecf/cbcsl/data100b/Chenqi/gan_results_for_presentation/humanBehavior/'



def dealData(srcDir, ganExp, nameFlag):
    for (dirpath, dirnames, filenames) in os.walk(srcDir+ganExp):
        if len(dirnames) == 1:
            srcNNFolder = srcDir+ganExp+dirnames[0]+'/NNmatchResult/'
            print("------------------deal with---------------------")
            print(srcNNFolder)
            assert(os.path.exists(srcNNFolder))
            # get the first 100 images:
            for (dirpath2, dirnames2, filenames2) in os.walk(srcNNFolder):
                assert(len(filenames2)>0)
                select_imgs = filenames2[:n_img]
                for (imgIdx, select_img) in enumerate(select_imgs):
                    assert(".jpg" in select_img or ".png" in select_img)
                    oldImgName = srcNNFolder + select_img
                    if 'resume' in ganExp:
                        ganExp = ganExp.split('_resume')[0] + '/'
                    if 'sub' not in ganExp and 'train' not in ganExp:
                        ganExp = ganExp.split('/')[0] + '_train/'
                    newImgName = nameFlag + ganExp.split('128_')[0] + ganExp.split('128_')[-1].split('/')[0] + '_' + str(imgIdx) + '.png'
                    # check img dim:
                    this_img = cv2.imread(oldImgName)
                    (img_height, img_width, ch) = this_img.shape
                    if img_height > 128:
                        this_img = this_img[2:,2:258]
                    img_gan = this_img[:,:128,:]
                    img_nn = this_img[:,128:,:]
                    cv2.imwrite(dstRootDir+'gan_generated/'+newImgName, img_gan)
                    cv2.imwrite(dstRootDir+'NN_match/'+newImgName, img_nn)
                    
    return



if __name__ == '__main__':
    #"""
    srcDir_stylegan = '/eecf/cbcsl/data100b/Chenqi/stylegan2/imgs/NN_query/'
    nameFlag2 = 'stylegan2_'
    for styleganExp in styleganExp_list:
        dealData(srcDir_stylegan, styleganExp, nameFlag2)
    #"""
    """
    srcDir_biggan = '/eecf/cbcsl/data100b/Chenqi/gan_results_for_presentation/biggan/NN_query/'
    nameFlag1 = 'biggan_'
    for bigganExp in bigganExp_list:
        dealData(srcDir_biggan, bigganExp, nameFlag1)
    """
    


            
            
            
            