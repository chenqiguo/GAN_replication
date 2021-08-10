#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 20:20:04 2020

@author: guo.1648
"""

# referenced from NN_getDist_testCode_forBiggan.py.

# using the NNmatchDist_threshold_value (10000) that we found,
# compute the number of replication cases (i.e., cases whose NN L-2 dist is
# greater than this threshold value).
# we will record the percent of replication cases (i.e., num_rep/sample_num)

# NOT run this! Run NN_getRepThreshPairImg_testCode_forBiggan.py instead!


import cv2
import os
import numpy as np
#from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestNeighbors
#from skimage import img_as_ubyte
#import torchvision.transforms as transforms

NNmatchDist_threshold_value = 8000 #9000 #10000

"""
#### for FLOWER_128_sub1000: 1000 images dataset
src_sampleSheetImg = '/scratch/BigGAN-PyTorch/samples/BigGAN_FLOWER_128_sub1000_BigGANdeep_seed0_Gch128_Dch128_Gd2_Dd2_bs16_nDa64_nGa64_Glr1.0e-04_Dlr4.0e-04_Gnlinplace_relu_Dnlinplace_relu_Ginitortho_Dinitortho_Gattn64_Dattn64_Gshared_hier_ema/fixed_samples38800.jpg'
srcRootDir_originDataImg = '/scratch/BigGAN-PyTorch/imgs/flower_sub1000/'

#dstRootDir_viewSampleSheetImgs = '/scratch/BigGAN-PyTorch/imgs/NN_query/FLOWER_128_sub1000/samples38800/view_sampleSheetImgs/'
#dstRootDir_NNmatchResult = '/scratch/BigGAN-PyTorch/imgs/NN_query/FLOWER_128_sub1000/samples38800/NNmatchResult/'
#dstImgName_NNmatchSheet = '/scratch/BigGAN-PyTorch/imgs/NN_query/FLOWER_128_sub1000/samples38800/NNmatchResultSheet.jpg'
#dstTxtName_matchDist = '/scratch/BigGAN-PyTorch/imgs/NN_query/FLOWER_128_sub1000/samples38800/NNmatchDist.txt'
dstTxtName_matchDistThresh = '/scratch/BigGAN-PyTorch/imgs/NN_query/FLOWER_128_sub1000/samples38800/NNmatchDist_smallerThanThresh.txt'

# parameters:
im_size = 128
batch_size = 16 # i.e., the sample sheet is of 4x4 !!!!:
num_row = 4
num_col = 4
"""
"""
#### for FLOWER_128: 8189 images dataset (the original FLOWER dataset)
src_sampleSheetImg = '/scratch/BigGAN-PyTorch/samples/BigGAN_FLOWER_128_BigGANdeep_seed0_Gch128_Dch128_Gd2_Dd2_bs48_nDa64_nGa64_Glr1.0e-04_Dlr4.0e-04_Gnlinplace_relu_Dnlinplace_relu_Ginitortho_Dinitortho_Gattn64_Dattn64_Gshared_hier_ema/fixed_samples21400.jpg'
srcRootDir_originDataImg = '/scratch/BigGAN-PyTorch/data/flower/jpg/'

#dstRootDir_viewSampleSheetImgs = '/scratch/BigGAN-PyTorch/imgs/NN_query/FLOWER_128/samples21400/view_sampleSheetImgs/'
#dstRootDir_NNmatchResult = '/scratch/BigGAN-PyTorch/imgs/NN_query/FLOWER_128/samples21400/NNmatchResult/'
#dstImgName_NNmatchSheet = '/scratch/BigGAN-PyTorch/imgs/NN_query/FLOWER_128/samples21400/NNmatchResultSheet.jpg'
#dstTxtName_matchDist = '/scratch/BigGAN-PyTorch/imgs/NN_query/FLOWER_128/samples21400/NNmatchDist.txt'
dstTxtName_matchDistThresh = '/scratch/BigGAN-PyTorch/imgs/NN_query/FLOWER_128/samples21400/NNmatchDist_smallerThanThresh.txt'

# parameters:
im_size = 128
batch_size = 48 # i.e., the sample sheet is of 6x8 !!!!:
num_row = 8
num_col = 6
"""

#### for CelebA_128_sub1000: 1000 images dataset
src_sampleSheetImg = '/scratch/BigGAN-PyTorch/samples/BigGAN_CelebA_128_sub1000_BigGANdeep_seed0_Gch128_Dch128_Gd2_Dd2_bs16_nDa64_nGa64_Glr1.0e-04_Dlr4.0e-04_Gnlinplace_relu_Dnlinplace_relu_Ginitortho_Dinitortho_Gattn64_Dattn64_Gshared_hier_ema/fixed_samples27000.jpg'
# newly modified: different from that in FLOWER_128_sub1000 and FLOWER_128:
#srcRootDir_originDataImg = '/scratch/BigGAN-PyTorch/data/flower/jpg/'
srcRootDir_imgNpz = '/scratch/BigGAN-PyTorch/CelebA_128_sub1000_imgs.npz'

#dstRootDir_viewSampleSheetImgs = '/scratch/BigGAN-PyTorch/imgs/NN_query/FLOWER_128/samples21400/view_sampleSheetImgs/'
#dstRootDir_NNmatchResult = '/scratch/BigGAN-PyTorch/imgs/NN_query/FLOWER_128/samples21400/NNmatchResult/'
#dstImgName_NNmatchSheet = '/scratch/BigGAN-PyTorch/imgs/NN_query/FLOWER_128/samples21400/NNmatchResultSheet.jpg'
#dstTxtName_matchDist = '/scratch/BigGAN-PyTorch/imgs/NN_query/FLOWER_128/samples21400/NNmatchDist.txt'
dstTxtName_matchDistThresh = '/scratch/BigGAN-PyTorch/imgs/NN_query/CelebA_128_sub1000/samples27000/NNmatchDist_smallerThanThresh.txt'

# parameters:
im_size = 128
batch_size = 16 # i.e., the sample sheet is of 4x4 !!!!:
num_row = 4
num_col = 4


def dealWith_sampleSheet():
    sampleSheet_img = cv2.imread(src_sampleSheetImg)
    (sheet_img_height, sheet_img_width, ch) = sampleSheet_img.shape
    
    single_img_height = sheet_img_height//num_row # 130
    single_img_width = sheet_img_width//num_col # 130
    
    # a list to store each image in the sampleSheet_img
    sample_img_list = []
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
    
    return sample_img_list


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
    
    # newly modified: different from that in FLOWER_128_sub1000 and FLOWER_128:
    """
    for (dirpath, dirnames, filenames) in os.walk(srcRootDir_originDataImg):
        for filename in filenames:
            if ".jpg" in filename:
    """
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


def query_NN_wrapper(sample_img_list):
    # this is a wrapper func!
    
    # first, get the training set from original images:
    len_featVec = len(image_to_feature_vector(sample_img_list[0]))
    dim = (sample_img_list[0].shape[1],sample_img_list[0].shape[0])
    trainSet_feats, all_origin_img_names = generateTrainSet(len_featVec, dim)
    
    neigh = NearestNeighbors(n_neighbors=1) # radius=0.4
    neigh.fit(trainSet_feats)
    
    # then, query:
    #match_img_list = []
    #match_distance_strs = ''
    match_distance_thresh_strs = ''
    match_num = 0
    for i in range(len(sample_img_list)):
        single_sample_img = sample_img_list[i]
        # get the query vector:
        single_sample_img_vec = image_to_feature_vector(single_sample_img)
        # NN to search:
        match_distance, match_idx = neigh.kneighbors([single_sample_img_vec], 1, return_distance=True)
        match_distance = match_distance[0][0]
        match_idx = match_idx[0][0]
        
        match_imgName = all_origin_img_names[match_idx].split('/')[-1]
        #match_img = trainSet_feats[match_idx,:].reshape((dim[1],dim[0],3))
        #match_img_list.append(match_img)
        # save the matching result:
        #im_h = cv2.hconcat([single_sample_img, match_img])
        #cv2.imwrite(dstRootDir_NNmatchResult+str(i+1)+'_'+match_imgName, im_h)
        # newly added: also save the corresponding match_distance into txt file:
        #match_distance_strs += str(i+1)+'_'+match_imgName + ': match_distance = ' + str(match_distance) + '\n'
        
        if match_distance <= NNmatchDist_threshold_value:
            match_num += 1
            match_distance_thresh_strs += str(i+1)+'_'+match_imgName + ': match_distance = ' + str(match_distance) + '\n'
        
    """
    # also combine the match_img together into a corresponding sheet!
    combine_matchingResult(match_img_list)
    # newly added: also save the corresponding match_distance into txt file:
    f = open(dstTxtName_matchDist, 'w')
    f.write(match_distance_strs)
    f.close()
    """
    match_percent = match_num / len(sample_img_list) * 100
    match_distance_thresh_strs = 'match_num = ' + str(match_num) + '\n' \
                                 + 'total sample_num = ' + str(len(sample_img_list)) + '\n' \
                                 + 'match_percent = ' + str(match_percent) + '%\n' \
                                 + match_distance_thresh_strs
    # newly added: save the values related to match_distance smaller than thresh into txt file:
    f = open(dstTxtName_matchDistThresh, 'w')
    f.write(match_distance_thresh_strs)
    f.close()
    
    return



if __name__ == '__main__':
    # first, deal with the sample sheet:
    sample_img_list = dealWith_sampleSheet()
    
    # finally, query each single_sample_img into original dataset (FLOWER_128_xxx here);
    # also, save the matching results:
    query_NN_wrapper(sample_img_list)



