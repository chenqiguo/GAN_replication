#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  8 13:18:52 2021

@author: guo.1648
"""

# my code to add fake (gan generated) images to the original dataset to train simCLR! <-- for my v2

import os, random, shutil


select_num = 300 # randomly selected this number of gan generated fake images for each stylegan2 experiment

dstDir = '/eecf/cbcsl/data100b/Chenqi/new_metrics/SimCLR/data/FLOWER_gan/jpg/'

srcDir1 = '/eecf/cbcsl/data100b/Chenqi/stylegan2/imgs/NN_query/FLOWER_128/fakes002526/view_sampleSheetImgs/'
srcDir2 = '/eecf/cbcsl/data100b/Chenqi/stylegan2/imgs/NN_query/FLOWER_128_sub1000_resume/fakes003248/view_sampleSheetImgs/'
srcDir3 = '/eecf/cbcsl/data100b/Chenqi/stylegan2/imgs/NN_query/FLOWER_128_sub4000_resume/fakes003248/view_sampleSheetImgs/'


srcDir_list = [srcDir1, srcDir2, srcDir3]
nameFlag_list = ['train_', 'sub1000_', 'sub4000_']


if __name__ == '__main__':
    for i in range(len(srcDir_list)):
        srcDir = srcDir_list[i]
        nameFlag = nameFlag_list[i]
        random_filenames = random.sample(os.listdir(srcDir), select_num)
        for fname in random_filenames:
            srcpath = os.path.join(srcDir, fname)
            destName = dstDir + 'gan_' + nameFlag + fname
            #print()
            shutil.copyfile(srcpath, destName)
        


