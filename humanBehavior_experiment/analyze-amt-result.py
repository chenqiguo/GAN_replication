#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 17:54:18 2021

@author: qianlifeng
"""

import csv
import numpy as np
import os

all_rows = []
with open('/Volumes/Samsung_T5/OSU/paper/GAN-replication/generated samples/Batch_4362813_batch_results.csv', newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',', quotechar='"')
    for row in spamreader:
        all_rows.append(row)
        
head = all_rows.pop(0)

csv_dict_list = []
for row in all_rows:
    csv_dict = {}
    for key,value in zip(head,row): 
        csv_dict[key] = value
    csv_dict_list.append(csv_dict)
    
all_url = []
all_label = []
for answer in csv_dict_list:
    url1 = answer['Input.image_url1']
    url2 = answer['Input.image_url2']
    url3 = answer['Input.image_url3']
    url4 = answer['Input.image_url4']
    url5 = answer['Input.image_url5']
    
    label1 = answer['Answer.category1']+answer['Answer.category1.label']
    label2 = answer['Answer.category2']+answer['Answer.category2.label']
    label3 = answer['Answer.category3']+answer['Answer.category3.label']
    label4 = answer['Answer.category4']+answer['Answer.category4.label']
    label5 = answer['Answer.category5']+answer['Answer.category5.label']
    
    all_url+=[url1,url2,url3,url4,url5]
    all_label+=[label1,label2,label3,label4,label5]
    
all_label_code = []
for this_label in all_label: 
    if this_label == 'Excellent':
        this_code = 5
    elif this_label == 'Good':
        this_code = 4
    elif this_label == 'Fair':
        this_code = 3
    elif this_label == 'Poor':
        this_code = 2
    elif this_label == 'Terrible':
        this_code = 1
    all_label_code.append(this_code)
all_label_np = np.array(all_label_code).astype(float)
    
all_img_result = []
unique_url = set(all_url)
for img_url in list(unique_url):
    result_dict = {}
    indices = [i for i, x in enumerate(all_url) if x == img_url]
    avg_code = np.mean(all_label_np[indices])
    
    img_type = img_url.split('_')[1].split('/')[0]
    gan = img_url.split('_')[1].split('/')[1]
    dataset = img_url.split('_')[2]
    subset = img_url.split('_')[3]
    image_id = img_url.split('_')[4]
    
    result_dict['type'] = img_type
    result_dict['gan'] = gan
    result_dict['dataset'] = dataset
    if subset != 'train':
        result_dict['subset'] = int(subset[3:]) 
    else:
        result_dict['subset'] = subset
    result_dict['image_id'] = image_id
    result_dict['avg_code'] = avg_code
    
    all_img_result.append(result_dict)
    
target_gan = 'biggan'
target_dataset = 'CelebA'
target_type = 'generated'

subset_results = {}
unique_subset = list(set([s['subset'] for s in all_img_result]))
subset_results = {subset: [] for subset in unique_subset} 
for result in all_img_result:
    if result['type'] == target_type and result['dataset'] == target_dataset and result['gan'] == target_gan:
        subset_results[result['subset']] += [result['avg_code']]
            
subset_mean = {subset: np.mean(np.array(code)) for subset,code in subset_results.items()} 

x = [subset for subset,code in subset_mean.items() if np.isnan(code) == False]
y = [code for subset,code in subset_mean.items() if np.isnan(code) == False]

idx = np.argsort(x)

x = np.array(x)[idx]
y = np.array(y)[idx]

import matplotlib.pyplot as plot
plot.plot(x,y)