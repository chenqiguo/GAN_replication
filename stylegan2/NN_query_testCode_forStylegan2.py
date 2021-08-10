#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 20 15:33:56 2020

@author: guo.1648
"""

# referenced from NN_query_testCode_forBiggan.py in biggan.
# This code is for stylegan2.

# my testing code:
# Try the NN with the generated images as query, the original dataset as training set,
# to see if the NN can find original images that we found similar to the generated images.

# Note: this code corresponds with FLOWER_128, FLOWER_128_sub1000, and FLOWER_128_sub4000

# NOT run this! run NN_getDist_testCode_forStylegan2.py instead.

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

dstRootDir_viewSampleSheetImgs = '/eecf/cbcsl/data100b/Chenqi/stylegan2/imgs/NN_query/FLOWER_128/view_sampleSheetImgs/'
dstRootDir_NNmatchResult = '/eecf/cbcsl/data100b/Chenqi/stylegan2/imgs/NN_query/FLOWER_128/NNmatchResult/'
dstImgName_NNmatchSheet = '/eecf/cbcsl/data100b/Chenqi/stylegan2/imgs/NN_query/FLOWER_128/NNmatchResultSheet.png'
"""
"""
#### for FLOWER_128_sub1000: 1000 images dataset (resume)
src_sampleSheetImg = '/eecf/cbcsl/data100b/Chenqi/stylegan2/results/results_FLOWER_128_sub1000_resume/00000-stylegan2-FLOWER_128_sub1000-1gpu-config-f/fakes003248.png'
srcRootDir_originDataImg = '/eecf/cbcsl/data100b/Chenqi/stylegan2/datasets_images/FLOWER_128_sub1000/' # these images are generated from tfrecords using code mycode_loadImgFromTFrecords.py

dstRootDir_viewSampleSheetImgs = '/eecf/cbcsl/data100b/Chenqi/stylegan2/imgs/NN_query/FLOWER_128_sub1000_resume/view_sampleSheetImgs/'
dstRootDir_NNmatchResult = '/eecf/cbcsl/data100b/Chenqi/stylegan2/imgs/NN_query/FLOWER_128_sub1000_resume/NNmatchResult/'
dstImgName_NNmatchSheet = '/eecf/cbcsl/data100b/Chenqi/stylegan2/imgs/NN_query/FLOWER_128_sub1000_resume/NNmatchResultSheet.png'
"""

#### for FLOWER_128_sub4000: 4000 images dataset (resume)
src_sampleSheetImg = '/eecf/cbcsl/data100b/Chenqi/stylegan2/results/results_FLOWER_128_sub4000_resume/00000-stylegan2-FLOWER_128_sub4000-1gpu-config-f/fakes003248.png'
srcRootDir_originDataImg = '/eecf/cbcsl/data100b/Chenqi/stylegan2/datasets_images/FLOWER_128_sub4000/' # these images are generated from tfrecords using code mycode_loadImgFromTFrecords.py

dstRootDir_viewSampleSheetImgs = '/eecf/cbcsl/data100b/Chenqi/stylegan2/imgs/NN_query/FLOWER_128_sub4000_resume/view_sampleSheetImgs/'
dstRootDir_NNmatchResult = '/eecf/cbcsl/data100b/Chenqi/stylegan2/imgs/NN_query/FLOWER_128_sub4000_resume/NNmatchResult/'
dstImgName_NNmatchSheet = '/eecf/cbcsl/data100b/Chenqi/stylegan2/imgs/NN_query/FLOWER_128_sub4000_resume/NNmatchResultSheet.png'



# parameters:
im_size = 128
# note: the sample sheet is of 32x32:
num_row = 32
num_col = 32


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
            if ".jpg" in filename:
                #print("------------------deal with---------------------")
                #print(filename)
                origin_img = cv2.imread(srcRootDir_originDataImg+filename)
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
    for i in range(len(sample_img_list)):
        single_sample_img = sample_img_list[i]
        # get the query vector:
        single_sample_img_vec = image_to_feature_vector(single_sample_img)
        # NN to search:
        match_idx = neigh.kneighbors([single_sample_img_vec], 1, return_distance=False)[0][0]
        match_imgName = all_origin_img_names[match_idx]
        match_img = trainSet_feats[match_idx,:].reshape((dim[1],dim[0],3))
        match_img_list.append(match_img)
        # save the matching result:
        im_h = cv2.hconcat([single_sample_img, match_img])
        cv2.imwrite(dstRootDir_NNmatchResult+str(i+1)+'_'+match_imgName, im_h)
        
    # newly added: also combine the match_img together into a corresponding sheet!
    combine_matchingResult(match_img_list)
    
    return


if __name__ == '__main__':
    #"""
    # first, deal with the sample sheet:
    sample_img_list = dealWith_sampleSheet()
    # for debug: save the generated sample images to visualize:
    for i in range(len(sample_img_list)):
        single_sample_img = sample_img_list[i]
        cv2.imwrite(dstRootDir_viewSampleSheetImgs+str(i+1)+'.png', single_sample_img)
    #"""
    
    # finally, query each single_sample_img into original dataset (FLOWER_128_xxx here);
    # also, save the matching results:
    query_NN_wrapper(sample_img_list)

