#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 17:06:12 2021

@author: guo.1648
"""

# from all the intdim_k_repeated_dicts_sz32.pkl files,
# get the mean of intdim_k_repeated as the Intrinsic Dimension values!


import os
import pickle
import numpy as np


"""
srcRootDir = '/eecf/cbcsl/data100b/Chenqi/dataset_complexity/hist_intdim_mle/'
pklName = 'intdim_k_repeated_dicts_sz32.pkl' # sz32 sz128
pklName_MNIST = 'intdim_k_repeated_dicts_sz32_grayscale.pkl' # for MNIST dataset

folder_list = ['CelebA_128_sub200/', 'CelebA_128_sub600/', 'CelebA_128_sub1000/', 'CelebA_128_sub4000/', 'CelebA_128_sub8000/',
               'FLOWER_128_sub1000/', 'FLOWER_128_sub2000/', 'FLOWER_128_sub4000/', 'FLOWER_128_sub6000/', 'FLOWER_128/',
               'LSUN_128_sub200/', 'LSUN_128_sub1000/', 'LSUN_128_sub5000/', 'LSUN_128_sub10000/', 'LSUN_128_sub30000/']
#              'MNIST_128_sub10000/', 'MNIST_128_sub30000/', 'MNIST_128_train/'

dstPklName = 'Intrinsic_Dimension_values_sz32.pkl' #sz32 sz128
"""

# for rebuttal:

# CIFAR10 dataset:
srcRootDir = '/eecf/cbcsl/data100b/Chenqi/dataset_complexity/hist_intdim_mle/'
pklName = 'intdim_k_repeated_dicts_sz32.pkl' # sz32 sz128

folder_list = ['CIFAR10_32_sub1000/', 'CIFAR10_32_sub4000/', 'CIFAR10_32_sub8000/', 'CIFAR10_32_sub10000/']

dstPklName = 'Intrinsic_Dimension_values_sz32_CIFAR10.pkl' #sz32 sz128



if __name__ == '__main__':
    
    dict_list = []
    
    for folder in folder_list:
        if 'MNIST' in folder:
            srcPkl = srcRootDir + folder + pklName_MNIST
        else:
            srcPkl = srcRootDir + folder + pklName
        # load the pickle file:
        f_pkl = open(srcPkl,'rb')
        store_dict = pickle.load(f_pkl)
        f_pkl.close()
        
        if len(store_dict) == 1: # this means store_dict is a list:
            store_dict = store_dict[0]
        intdim_k_repeated = store_dict['intdim_k_repeated']
        
        #print()
        
        hist = intdim_k_repeated.mean(axis=1) # this is how we plot the hist
        # tmp = np.histogram(intdim_k_repeated.mean(axis=1))
        ID_val = np.mean(hist) # our final ID value (representing dataset complexity)
        
        # save to dict:
        dict_ = {'dataset': folder,
                 'ID_val': ID_val}
        dict_list.append(dict_)
        
    f_pkl = open(srcRootDir+dstPklName, 'wb')
    pickle.dump(dict_list, f_pkl)
    f_pkl.close()



