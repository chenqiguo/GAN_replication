#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 15 14:36:57 2021

@author: guo.1648
"""

# Plot histograms of the NN distance for each stylegan2 MNIST subset experiments,
# and stylegan2 flower subset experiments.

# This code is for checking if the NN distance (L2-norm) distribustion are the same.
# Here we use NNmatchDist_threshold_value = 8000

import matplotlib.pyplot as plt
import os
import numpy as np

"""
#### for MNIST_128_sub10000: 10000 images dataset
# for fakes005173:
srcRootDir = '/eecf/cbcsl/data100b/Chenqi/gan_results_for_presentation/stylegan2/NN_query/MNIST_128_sub10000/fakes005173/'
srcTxt = 'NNmatchDist.txt' #'NN_pltDistHist_thresh8000.txt'

dstPlt = 'MNIST_128_sub10000_NNdistHist.eps' #'MNIST_128_sub10000_NNdistHist_thresh8000.eps'
titleFlag = 'MNIST_128_sub10000'
"""
"""
#### for MNIST_128_sub30000: 30000 images dataset
# for fakes005053:
srcRootDir = '/eecf/cbcsl/data100b/Chenqi/gan_results_for_presentation/stylegan2/NN_query/MNIST_128_sub30000/fakes005053/'
srcTxt = 'NNmatchDist.txt' #'NN_pltDistHist_thresh8000.txt'

dstPlt = 'MNIST_128_sub30000_NNdistHist.eps' #'MNIST_128_sub30000_NNdistHist_thresh8000.eps'
titleFlag = 'MNIST_128_sub30000'
"""
"""
#### for MNIST_128_train: 60000 images dataset
# for fakes003609:
srcRootDir = '/eecf/cbcsl/data100b/Chenqi/gan_results_for_presentation/stylegan2/NN_query/MNIST_128_train/fakes003609/'
srcTxt = 'NNmatchDist.txt' #'NN_pltDistHist_thresh8000.txt'

dstPlt = 'MNIST_128_train_NNdistHist.eps' #'MNIST_128_train_NNdistHist_thresh8000.eps'
titleFlag = 'MNIST_128_train'
"""
"""
#### for MNIST
srcTxt = 'NNmatchDist.txt'
srcRootDir1 = '/eecf/cbcsl/data100b/Chenqi/gan_results_for_presentation/stylegan2/NN_query/MNIST_128_sub10000/fakes005173/'
leg1 = 'MNIST_128_sub10000'
srcRootDir2 = '/eecf/cbcsl/data100b/Chenqi/gan_results_for_presentation/stylegan2/NN_query/MNIST_128_sub30000/fakes005053/'
leg2 = 'MNIST_128_sub30000'
srcRootDir3 = '/eecf/cbcsl/data100b/Chenqi/gan_results_for_presentation/stylegan2/NN_query/MNIST_128_train/fakes003609/'
leg3 = 'MNIST_128_train'

figName = '/eecf/cbcsl/data100b/Chenqi/gan_results_for_presentation/stylegan2/NN_query/MNIST_128_3subsets_NNdistHist.png'
titleFlag = 'MNIST_128_3subsets'
"""


"""
#### for FLOWER_128_sub1000_resume: 1000 images dataset
# for fakes003248:
srcRootDir = '/eecf/cbcsl/data100b/Chenqi/gan_results_for_presentation/stylegan2/NN_query/FLOWER_128_sub1000_resume/fakes003248/'
srcTxt = 'NNmatchDist.txt'

dstPlt = 'FLOWER_128_sub1000_resume_NNdistHist.eps'
titleFlag = 'FLOWER_128_sub1000_resume'
"""
"""
#### for FLOWER_128_sub4000_resume: 4000 images dataset
# for fakes003248:
srcRootDir = '/eecf/cbcsl/data100b/Chenqi/gan_results_for_presentation/stylegan2/NN_query/FLOWER_128_sub4000_resume/fakes003248/'
srcTxt = 'NNmatchDist.txt'

dstPlt = 'FLOWER_128_sub4000_resume_NNdistHist.eps'
titleFlag = 'FLOWER_128_sub4000_resume'
"""
"""
#### for FLOWER_128: ~8000 images dataset
# for fakes002526:
srcRootDir = '/eecf/cbcsl/data100b/Chenqi/gan_results_for_presentation/stylegan2/NN_query/FLOWER_128/fakes002526/'
srcTxt = 'NNmatchDist.txt'

dstPlt = 'FLOWER_128_NNdistHist.eps'
titleFlag = 'FLOWER_128_train'
"""

############################### USE BELOW:
"""
#### for FLOWER:
srcTxt = 'NNmatchDist.txt'
srcRootDir1 = '/eecf/cbcsl/data100b/Chenqi/gan_results_for_presentation/stylegan2/NN_query/FLOWER_128_sub1000_resume/fakes003248/'
leg1 = 'FLOWER_128_sub1000_resume'
srcRootDir2 = '/eecf/cbcsl/data100b/Chenqi/gan_results_for_presentation/stylegan2/NN_query/FLOWER_128_sub4000_resume/fakes003248/'
leg2 = 'FLOWER_128_sub4000_resume'
srcRootDir3 = '/eecf/cbcsl/data100b/Chenqi/gan_results_for_presentation/stylegan2/NN_query/FLOWER_128/fakes002526/'
leg3 = 'FLOWER_128_train'

figName = '/eecf/cbcsl/data100b/Chenqi/gan_results_for_presentation/stylegan2/NN_query/FLOWER_128_3subsets_NNdistHist.png'
titleFlag = 'FLOWER_128_3subsets'
"""

#### for FLOWER simCLR_v2:
srcTxt = 'NNmatchDist.txt'
srcRootDir1 = '/eecf/cbcsl/data100b/Chenqi/gan_results_for_presentation/stylegan2/NNquery_simCLR_v2/FLOWER_128_sub1000_resume/fakes003248/'
leg1 = 'FLOWER_128_sub1000_resume'
srcRootDir2 = '/eecf/cbcsl/data100b/Chenqi/gan_results_for_presentation/stylegan2/NNquery_simCLR_v2/FLOWER_128_sub4000_resume/fakes003248/'
leg2 = 'FLOWER_128_sub4000_resume'
srcRootDir3 = '/eecf/cbcsl/data100b/Chenqi/gan_results_for_presentation/stylegan2/NNquery_simCLR_v2/FLOWER_128/fakes002526/'
leg3 = 'FLOWER_128_train'

figName = '/eecf/cbcsl/data100b/Chenqi/gan_results_for_presentation/stylegan2/NNquery_simCLR_v2/FLOWER_128_3subsets_simCLR_NNdistHist.png'
titleFlag = 'FLOWER_128_3subsets_simCLR_NNdist'



def loadDistFromTxt(txtName):
    dist_list = []
    with open(txtName, 'r') as f:
        for line in f.readlines():
            #print(line)
            line = line.strip()
            #print(line)
            if 'match_distance = ' in line:
                #print('here!!!')
                this_dist = line.split('match_distance = ')[-1]
                this_dist = float(this_dist)
                #print(this_dist)
                dist_list.append(this_dist)
    
    return dist_list



if __name__ == "__main__":
    """
    ### Plot each subset seperately:
    # First, get the NN distance from the txt file:
    txtName = srcRootDir + srcTxt
    dist_list = loadDistFromTxt(txtName)
    # Next, build and save the histogram plt:
    figName = srcRootDir + dstPlt
    fig1 = plt.figure()
    plt.hist(dist_list, bins=16)
    plt.title(titleFlag + "_NNdistHist") # "_NNdistHist_thresh8000"
    fig1.savefig(figName)
    """
    
    #print()
    
    
    ### Plot 3 subsets together:
    # First, get the NN distance from the txt file:
    dist_list1 = loadDistFromTxt(srcRootDir1+srcTxt)
    dist_list2 = loadDistFromTxt(srcRootDir2+srcTxt)
    dist_list3 = loadDistFromTxt(srcRootDir3+srcTxt)
    # Next, build and save the histogram plt:
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.hist(dist_list1, bins=16, fc=(1, 0, 0, 0.5))
    ax.hist(dist_list2, bins=16, fc=(0, 1, 0, 0.5))
    ax.hist(dist_list3, bins=16, fc=(0, 0, 1, 0.5))
    plt.title(titleFlag + "_NNdistHist")
    ax.legend([leg1,leg2,leg3], loc=3) #0
    fig.savefig(figName)

