#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  1 13:23:04 2021

@author: guo.1648
"""

# My 1st try to use simCLR features to do NN query.

import argparse

from torchvision import transforms, datasets
from torch.utils.data import Dataset, DataLoader
import numpy as np

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from thop import profile, clever_format
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from tqdm import tqdm

import utils
from model import Model

from sklearn.neighbors import NearestNeighbors
import cv2


class NNquery_Net(nn.Module): # using h (feature) to do NN query for our gan results!
    def __init__(self, pretrained_path): # pretrained_path
        super(NNquery_Net, self).__init__()

        # encoder
        self.f = Model().f
        # classifier
        #self.fc = nn.Linear(2048, num_class, bias=True) # we do NOT need this here!
        self.load_state_dict(torch.load(pretrained_path, map_location='cpu'), strict=False)

    def forward(self, x):
        x = self.f(x)
        feature = torch.flatten(x, start_dim=1)
        #out = self.fc(feature)
        #return out
        return F.normalize(feature, dim=-1) #feature # should do normalization as in model.py !!!! <-- modify it!




def get_mean_std_forDataset(data_dir,img_size,batch_size,isGray):
    # newly added: compute the mean and std for transforms.Normalize using whole dataset:
    tmp_data = datasets.ImageFolder(root=data_dir, transform=transforms.Compose([transforms.Resize(img_size),
                                                                                 transforms.CenterCrop(img_size),
                                                                                 transforms.ToTensor()]))
    tmp_loader = DataLoader(tmp_data, batch_size=batch_size, shuffle=False, num_workers=16)
    
    mean = 0.
    std = 0.
    nb_samples = 0.
    if not isGray:
        for data, _ in tmp_loader:
            batch_samples = data.size(0)
            data = data.view(batch_samples, data.size(1), -1)
            mean += data.mean(2).sum(0)
            std += data.std(2).sum(0)
            nb_samples += batch_samples
    #else: for MNIST
    
    mean /= nb_samples
    std /= nb_samples
    
    return (mean, std)


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


def NNquery_func(net, train_loader, test_loader):
    # do our NN query using the feature output by the model:
    net.eval()
    
    feature_bank = [] # the list of all training features
    all_origin_img_names = []
    with torch.no_grad():
        # generate feature bank
        #for data, _target in tqdm(train_loader, desc='Feature extracting'):
        for i, (data, _target) in enumerate(train_loader, 0):
            feature = net(data.cuda(non_blocking=True)) # of shape: torch.Size([batch_size, 2048])
            feature_bank.append(feature.cpu())
            origin_imgName, _ = train_loader.dataset.samples[i]
            all_origin_img_names.append(origin_imgName)
            
        # [D, N]
        #print()
        feature_bank = torch.cat(feature_bank, dim=0).t().contiguous() # of shape torch.Size([2048, train_set_size])
        # convert tensor to numpy array:
        feature_bank_arr = torch.Tensor.cpu(feature_bank).detach().numpy() # of shape (2048, train_set_size)
        # transpose this array:
        feature_bank_arr = np.transpose(feature_bank_arr) # of shape (train_set_size, 2048)
        
        neigh = NearestNeighbors(n_neighbors=1) # radius=0.4
        neigh.fit(feature_bank_arr)
        
        # loop test data to predict the label by 1-nn search
        match_img_list = []
        f = open(result_dir+"NNmatchDist.txt", "w")
        for i, (data, _target) in enumerate(test_loader, 0): # only 1 img at a time (since batch_size=1)
            feature = net(data.cuda(non_blocking=True))
            gan_imgName, _ = test_loader.dataset.samples[i]
            #print()
            feature_arr = torch.Tensor.cpu(feature).detach().numpy()[0] # of shape (2048,)
            match_distance, match_idx = neigh.kneighbors([feature_arr], 1, return_distance=True)
            match_distance = match_distance[0][0]
            match_idx = match_idx[0][0]
            # save the matched pair of images:
            matchOrig_imgName = all_origin_img_names[match_idx]
            matchOrig_img = cv2.imread(matchOrig_imgName)
            gan_img = cv2.imread(gan_imgName)
            im_h = cv2.hconcat([gan_img, matchOrig_img])
            
            result_imgName = gan_imgName.split('/')[-1].split('.')[0] + '_' + matchOrig_imgName.split('/')[-1].split('.')[0] + '.jpg'
            cv2.imwrite(result_dir+'NNmatchResult/'+result_imgName, im_h)
                
            f.write(result_imgName + ': match_distance = ' + str(match_distance) + '\n')
            
            match_img_list.append(matchOrig_img)
            
        f.close()
        
        # also combine the match_img together into a corresponding sheet!
        combine_matchingResult(match_img_list)
        
        



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='NNquery SimCLR')
    parser.add_argument('--model_path', type=str, default='results/FLOWER_128/best_128_0.5_200_26_2000_model.pth',
                        help='The pretrained model path')
    parser.add_argument('--dataset', default='FLOWER_128_train', type=str, help='Name of the corresponding gan experiment, eg, FLOWER_128_train')
    parser.add_argument('--data_dir', default='/eecf/cbcsl/data100b/Chenqi/stylegan2/datasets_images/FLOWER_128/', type=str, help='Dir of the original training dataset')
    parser.add_argument('--gan_dir', default='/eecf/cbcsl/data100b/Chenqi/stylegan2/imgs/NNquery_simCLR/FLOWER_128/fakes002526/view_sampleSheetImgs/', type=str, help='Dir of the gan generated dataset, as testing')
    #parser.add_argument('--label_file', default='/eecf/cbcsl/data100b/Chenqi/data/flower_labels.txt', type=str, help='Path to the txt file with class labels')
    #parser.add_argument('--batch_size', default=26, type=int, help='Number of images in each mini-batch')
    parser.add_argument('--result_dir', default='/eecf/cbcsl/data100b/Chenqi/stylegan2/imgs/NNquery_simCLR/FLOWER_128/fakes002526/', type=str, help='Dir of the matching results')
    parser.add_argument('--num_row', default=32, type=int, help='Number of rows for the NN match result sheet')
    parser.add_argument('--num_col', default=32, type=int, help='Number of columns for the NN match result sheet')
    parser.add_argument('--mean_std_data_dir', default='/eecf/cbcsl/data100b/Chenqi/data/flower/', type=str, help='Dir of the dataset same as training in simclr, to compute mean & std')
    parser.add_argument('--old_batch_size', default=26, type=int, help='Number of images in each mini-batch same as training in simclr, to compute mean & std')
    
    
    # args parse
    args = parser.parse_args()
    model_path = args.model_path
    dataset = args.dataset
    img_size = int(dataset.split('_')[1])
    data_dir = args.data_dir
    gan_dir = args.gan_dir
    #label_file = args.label_file
    #batch_size = args.batch_size 
    batch_size = 1 # this is only used for dataset images loading, and 1 is convenient in NN query!
    result_dir = args.result_dir
    num_row = args.num_row
    num_col = args.num_col
    mean_std_data_dir = args.mean_std_data_dir
    old_batch_size = args.old_batch_size
    
    # initiate data transformation:
    if 'MNIST' not in dataset:
        # compute the mean and std for transforms.Normalize using whole original training dataset:
        img_means, img_stds = get_mean_std_forDataset(mean_std_data_dir,img_size,old_batch_size,isGray=False)
        NNquery_transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(img_means, img_stds)]) # ([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]) for cifar10
        #invTrans = transforms.Normalize(mean = -img_means/img_stds, std = 1/img_stds)
    # else:... (for MNIST)
    
    
    # get datasets:
    train_data = datasets.ImageFolder(root=data_dir, transform=NNquery_transform) # i.e., the original dataset, as training
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True)
    test_data = datasets.ImageFolder(root=gan_dir, transform=NNquery_transform) # i.e., the gan generated dataset, as testing
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True)
    
    
    # load model:
    model = NNquery_Net(pretrained_path=model_path).cuda()
    for param in model.f.parameters():
        param.requires_grad = False
    
    # do our NN query using the feature output by the model:
    NNquery_func(model, train_loader, test_loader)
    
    
    
    
    