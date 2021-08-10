#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 19 23:18:41 2020

@author: guo.1648
"""


import numpy as np
import tensorflow as tf
import dnnlib
import dnnlib.tflib as tflib
from dnnlib.tflib.autosummary import autosummary

from training import dataset
from training import misc
from metrics import metric_base
from dnnlib import EasyDict
import cv2


tf_config = {'rnd.np_random_seed': 1000}
tflib.init_tf(tf_config)
dataset_args = EasyDict(tfrecord_dir='CIFAR10_32_sub10000/') # modify this each time!: FLOWER_128_sub1000; FLOWER_128_sub4000; FLOWER_128; CelebA_128_sub1000; CelebA_128_sub4000
# fet the tfrecordDataset object
training_set = dataset.load_dataset(data_dir=dnnlib.convert_path('datasets/'), verbose=True, **dataset_args)

#grid_args = EasyDict(size='8k', layout='random')
#grid_size, grid_reals, grid_labels = misc.setup_snapshot_image_grid(training_set, **grid_args)

num_images = 10000 # modify this each time!: 1000; 4000; 8189
dstImg_dir = '/eecf/cbcsl/data100b/Chenqi/stylegan2/datasets_images/CIFAR10_32_sub10000/' # modify this each time!: FLOWER_128_sub1000; FLOWER_128_sub4000; FLOWER_128; CelebA_128_sub1000; CelebA_128_sub4000

# Newly added: only used for MNIST dataset:
# binarize the images!
#biFlag = True # for MNIST dataset
biFlag = False # for other (RGB or grayscale) dataset

for image_idx in range(num_images):
    print('Loading image %d/%d ...' % (image_idx, num_images))
    images, _labels = training_set.get_minibatch_np(1)
    #images = misc.adjust_dynamic_range(images, [0, 255], [-1, 1])
    #images = images[0,:,:,:]
    num, img_w, img_h = images.shape[0], images.shape[-1], images.shape[-2]
    assert(num==1)
    this_image = images[0]
    dstImgName = dstImg_dir + str(image_idx) + '.png'
    # newly added:
    if biFlag:
        _,this_image = cv2.threshold(this_image,127,255,cv2.THRESH_BINARY)
    
    misc.convert_to_pil_image(this_image, drange=training_set.dynamic_range).save(dstImgName)
    #print()
    # for debug: view this image
    #cv2.imwrite('/home/guo.1648/Desktop/images.png',images)
    




"""
tf.parse_single_example(training_set, {
        'data': tf.FixedLenFeature([], tf.uint8),
        'shape': tf.FixedLenFeature([], tf.int64),
    })
"""

"""
my_tfrecord_file = "/eecf/cbcsl/data100b/Chenqi/stylegan2/datasets/FLOWER_128/FLOWER_128-r07.tfrecords"
example = tf.train.Example()
i = 0
for record in tf.python_io.tf_record_iterator(my_tfrecord_file): # 1000/4000/8189 images saved here!
    #print(tf.train.Example.FromString(record))
    example.ParseFromString(record)
    f = example.features.feature
    data = f['data'].bytes_list.value[0]
    shape = (f['shape'].int64_list.value[0], f['shape'].int64_list.value[1], f['shape'].int64_list.value[2])
    
    #print()
    i += 1 # 1000 for sub1000! 4000 for sub4000! 8189 for all!
"""


