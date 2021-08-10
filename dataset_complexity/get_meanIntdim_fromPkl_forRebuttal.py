#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 16 23:07:45 2021

@author: guo.1648
"""

# code for rebuttal:
# from all the intdim_k_repeated_dicts_sz32.pkl files,
# get the mean of intdim_k_repeated as the Intrinsic Dimension values!

# referenced from get_meanIntdim_fromPkl.py


import os
import pickle
import numpy as np


# FLOWER dataset:
srcRootDir = '/eecf/cbcsl/data100b/Chenqi/dataset_complexity/hist_intdim_mle_forRebuttal/'
pklName = 'intdim_k_repeated_dicts_sz32.pkl' # sz32 sz128

k12_folder = 'k1_20_k2_30/'
folder_list = ['FLOWER_128_sub1000/', 'FLOWER_128_sub2000/', 'FLOWER_128_sub4000/', 'FLOWER_128_sub6000/', 'FLOWER_128/']

dstPklName = 'Intrinsic_Dimension_values_sz32_k1_20_k2_30_FLOWER.pkl' #sz32 sz128


if __name__ == '__main__':
    
    dict_list = []
    
    for folder in folder_list:
        srcPkl = srcRootDir + folder + k12_folder + pklName
        
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
        
    
    

