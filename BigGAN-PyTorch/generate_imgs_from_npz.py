#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 14 23:58:57 2021

@author: guo.1648
"""

# generate training images from npz files


import cv2
import os
import numpy as np


srcRootDir = '/eecf/cbcsl/data100b/Chenqi/BigGAN-PyTorch/'
srcFile = srcRootDir + 'FLOWER_128_imgs.npz'

dstRootDir = '/eecf/cbcsl/data100b/Chenqi/BigGAN-PyTorch/imgs/FLOWER_128/jpg/'


dim = (128,128)




def my_center_crop(origin_img, crop_size):
    y,x,_ = origin_img.shape
    startx = x//2-(crop_size//2)
    starty = y//2-(crop_size//2)    
    origin_img_centCrop = origin_img[starty:starty+crop_size,startx:startx+crop_size]
    
    return origin_img_centCrop



if __name__ == '__main__':
    images_arr = np.load(srcFile)
    images_list = list(images_arr['imgs'][:,0])
    
    for filename in images_list:
        print("------------------deal with---------------------")
        print(filename)
        origin_img = cv2.imread(filename)
        origin_img_centCrop = my_center_crop(origin_img, min(origin_img.shape[0],origin_img.shape[1]))
        # resize using linear interpolation:
        origin_img_centCrop_resize = cv2.resize(origin_img_centCrop, dim)
        
        newImgName = dstRootDir + filename.split('/')[-1]
        cv2.imwrite(newImgName, origin_img_centCrop_resize)
    


