#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 14 19:53:21 2021

@author: guo.1648
"""

# for rebuttal:
# referenced from NNquery_inceptionv3_myTest.py and NN_query_thresh_finalVer.py

# based on paper Image2StyleGAN,
# use a weighted combination of the inceptionv3 MSE loss and the pixel-wise MSE loss
# to compute the 1NN match.


import argparse

from torchvision import transforms, datasets
from torch.utils.data import Dataset, DataLoader
import numpy as np
from scipy import linalg # For numpy FID
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter as P
from torchvision.models.inception import inception_v3

from sklearn.neighbors import NearestNeighbors
import cv2
import os
from shutil import copyfile



class WrapInception(nn.Module):
    def __init__(self, net):
        super(WrapInception,self).__init__()
        self.net = net
        self.mean = P(torch.tensor([0.485, 0.456, 0.406]).view(1, -1, 1, 1),
                      requires_grad=False)
        self.std = P(torch.tensor([0.229, 0.224, 0.225]).view(1, -1, 1, 1),
                     requires_grad=False)
    def forward(self, x):
        # Normalize x
        x = (x + 1.) / 2.0
        x = (x - self.mean) / self.std
        # Upsample if necessary
        if x.shape[2] != 299 or x.shape[3] != 299:
          x = F.interpolate(x, size=(299, 299), mode='bilinear', align_corners=True)
        # 299 x 299 x 3
        x = self.net.Conv2d_1a_3x3(x)
        # 149 x 149 x 32
        x = self.net.Conv2d_2a_3x3(x)
        # 147 x 147 x 32
        x = self.net.Conv2d_2b_3x3(x)
        # 147 x 147 x 64
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # 73 x 73 x 64
        x = self.net.Conv2d_3b_1x1(x)
        # 73 x 73 x 80
        x = self.net.Conv2d_4a_3x3(x)
        # 71 x 71 x 192
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # 35 x 35 x 192
        x = self.net.Mixed_5b(x)
        # 35 x 35 x 256
        x = self.net.Mixed_5c(x)
        # 35 x 35 x 288
        x = self.net.Mixed_5d(x)
        # 35 x 35 x 288
        x = self.net.Mixed_6a(x)
        # 17 x 17 x 768
        x = self.net.Mixed_6b(x)
        # 17 x 17 x 768
        x = self.net.Mixed_6c(x)
        # 17 x 17 x 768
        x = self.net.Mixed_6d(x)
        # 17 x 17 x 768
        x = self.net.Mixed_6e(x)
        # 17 x 17 x 768
        # 17 x 17 x 768
        x = self.net.Mixed_7a(x)
        # 8 x 8 x 1280
        x = self.net.Mixed_7b(x)
        # 8 x 8 x 2048
        x = self.net.Mixed_7c(x)
        # 8 x 8 x 2048
        pool = torch.mean(x.view(x.size(0), x.size(1), -1), 2)
        # 1 x 1 x 2048
        logits = self.net.fc(F.dropout(pool, training=False).view(pool.size(0), -1))
        # 1000 (num_classes)
        return pool, logits


# Load and wrap the Inception model
def load_inception_net(parallel=False):
    inception_model = inception_v3(pretrained=True, transform_input=False)
    inception_model = WrapInception(inception_model.eval()).cuda()
    if parallel:
        print('Parallelizing Inception module...')
        inception_model = nn.DataParallel(inception_model)
    return inception_model


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
    cv2.imwrite(result_dir+'NNmatchResultSheet.png', match_img_sheet)
    
    return


def NNquery_func_comb(net, train_loader, test_loader, coeff_pix):
    # do our NN query using the feature output by the model combined with pixel-wised values
    net.eval()
    
    feature_bank = [] # newly modified: the list of all training features followed by pixel-wised values
    all_origin_img_names = []
    with torch.no_grad():
        # generate feature bank
        #for data, _target in tqdm(train_loader, desc='Feature extracting'):
        for i, (data, _target) in enumerate(train_loader, 0):
            print('************ train_loader: ' + str(i))
            
            pool_val, logits_val = net(data.float()) # pool_val.shape=torch.Size([1, 2048]); logits_val.shape=torch.Size([1, 1000])
            
            # newly added:
            # (1) NOT using coeff_pix version:
            img_flat = torch.flatten(data.float()).unsqueeze(0).cpu() # torch.Size([1,49152])
            incep_comb_pix = torch.cat((pool_val.cpu(),img_flat),1) # torch.Size([1, 51200])
            
            feature_bank.append(incep_comb_pix.cpu()) #.detach().numpy()
            origin_imgName, _ = train_loader.dataset.samples[i]
            all_origin_img_names.append(origin_imgName)
    
        #print()
        feature_bank = torch.cat(feature_bank, dim=0).t().contiguous() # of shape torch.Size([51200, train_set_size])
        # convert tensor to numpy array:
        feature_bank_arr = torch.Tensor.cpu(feature_bank).detach().numpy() # of shape (51200, train_set_size)
        # transpose this array:
        feature_bank_arr = np.transpose(feature_bank_arr) # of shape (train_set_size, 51200)
        
        neigh = NearestNeighbors(n_neighbors=1) # radius=0.4
        neigh.fit(feature_bank_arr)
        
        # loop test data to predict the label by 1-nn search
        match_img_list = []
        f = open(result_dir+"NNmatchDist.txt", "w")
        for i, (data, _target) in enumerate(test_loader, 0): # only 1 img at a time (since batch_size=1)
            print('************ test_loader: ' + str(i))
            
            pool_val, logits_val = net(data.float()) # pool_val.shape=torch.Size([1, 2048]); logits_val.shape=torch.Size([1, 1000])
            gan_imgName, _ = test_loader.dataset.samples[i]
            #print()
            
            # newly added:
            # (1) NOT using coeff_pix version:
            img_flat = torch.flatten(data.float()).unsqueeze(0).cpu() # torch.Size([1,49152])
            incep_comb_pix = torch.cat((pool_val.cpu(),img_flat),1) # torch.Size([1, 51200])
            
            feature_arr = torch.Tensor.cpu(incep_comb_pix).detach().numpy()[0] # of shape (51200,)
            
            match_distance, match_idx = neigh.kneighbors([feature_arr], 1, return_distance=True)
            match_distance = match_distance[0][0]
            match_idx = match_idx[0][0]
            # save the matched pair of images:
            matchOrig_imgName = all_origin_img_names[match_idx]
            matchOrig_img = cv2.imread(matchOrig_imgName)
            gan_img = cv2.imread(gan_imgName)
            
            if gan_img.shape[0] > 128:
                gan_img = gan_img[2:,2:,:]
            im_h = cv2.hconcat([gan_img, matchOrig_img])
            
            result_imgName = gan_imgName.split('/')[-1].split('.')[0] + '_' + matchOrig_imgName.split('/')[-1].split('.')[0] + '.jpg'
            cv2.imwrite(result_dir+'NNmatchResult/'+result_imgName, im_h)
                
            f.write(result_imgName + ': match_distance = ' + str(match_distance) + '\n')
            
            match_img_list.append(matchOrig_img)
            
        f.close()
        
        # also combine the match_img together into a corresponding sheet!
        combine_matchingResult(match_img_list)
    
    return



if __name__ == '__main__':
    # NOTE: since inceptionv3 space is specifically defined with imagenet pertrain,
    # here we only use the pretrained on, but NOT train it on our own dataset!
    # we will conduct the whole procedure for biggan & stylegan2 on FLOWER_128_sub1000 respectively.
    
    parser = argparse.ArgumentParser(description='NNquery inceptionv3 combine pixel-wise')
    parser.add_argument('--dataset', default='FLOWER_128', type=str, help='Name of the corresponding gan experiment, eg, FLOWER_128_sub1000')
    parser.add_argument('--data_dir', default='/eecf/cbcsl/data100b/Chenqi/stylegan2/datasets_images/FLOWER_128/', type=str, help='Dir of the original training dataset')
    parser.add_argument('--gan_dir', default='/eecf/cbcsl/data100b/Chenqi/gan_results_for_presentation/stylegan2/NNquery_inceptionv3_pixelwise/FLOWER_128/fakes002526/view_sampleSheetImgs/', type=str, help='Dir of the gan generated dataset, as testing')
    parser.add_argument('--result_dir', default='/eecf/cbcsl/data100b/Chenqi/gan_results_for_presentation/stylegan2/NNquery_inceptionv3_pixelwise/FLOWER_128/fakes002526/', type=str, help='Dir of the matching results')
    parser.add_argument('--num_row', default=32, type=int, help='Number of rows for the NN match result sheet')
    parser.add_argument('--num_col', default=32, type=int, help='Number of columns for the NN match result sheet')
    #parser.add_argument('--mean_std_data_dir', default='/eecf/cbcsl/data100b/Chenqi/data/flower/', type=str, help='Dir of the dataset same as training in simclr, to compute mean & std')
    #parser.add_argument('--old_batch_size', default=26, type=int, help='Number of images in each mini-batch same as training in simclr, to compute mean & std')
    
    # args parse
    args = parser.parse_args()
    dataset = args.dataset
    img_size = int(dataset.split('_')[1])
    data_dir = args.data_dir
    gan_dir = args.gan_dir
    batch_size = 1 # this is only used for dataset images loading, and 1 is convenient in NN query!
    result_dir = args.result_dir
    num_row = args.num_row
    num_col = args.num_col
    #mean_std_data_dir = args.mean_std_data_dir
    #old_batch_size = args.old_batch_size
    
    # initiate data transformation:
    if 'MNIST' not in dataset:
        # compute the mean and std for transforms.Normalize using whole original training dataset:
        #img_means, img_stds = get_mean_std_forDataset(mean_std_data_dir,img_size,old_batch_size,isGray=False)
        NNquery_transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.CenterCrop(img_size),
            transforms.ToTensor()]) # transforms.Normalize(img_means, img_stds) <-- NO need
        #invTrans = transforms.Normalize(mean = -img_means/img_stds, std = 1/img_stds)
    # else:... (for MNIST)
    
    # get datasets:
    train_data = datasets.ImageFolder(root=data_dir, transform=NNquery_transform) # i.e., the original dataset, as training
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True)
    test_data = datasets.ImageFolder(root=gan_dir, transform=NNquery_transform) # i.e., the gan generated dataset, as testing
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True)
    
    # load model:
    #inception_model = load_inception_net(parallel=True) # just for debug: comment out it!
    """
    for param in inception_model.f.parameters():
        param.requires_grad = False
    """
    
    # newly added: compute weight for pixel-wise NN:
    if 'MNIST' not in dataset: # for 3-channel images
        N_ = 3*img_size*img_size
        weight_pix = 1.0 / N_
        import math
        coeff_pix = math.sqrt(weight_pix)
    
    # newly modified:
    # do our NN query using the feature output by the model combined with pixel-wised values:
    NNquery_func_comb(inception_model, train_loader, test_loader, coeff_pix)
    
    


