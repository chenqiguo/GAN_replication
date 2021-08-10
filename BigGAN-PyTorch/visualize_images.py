#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  2 05:41:52 2020

Visual checking the result of the subsampling 

@author: feng.559
"""

import glob as glob
import numpy as np
import os, shutil


for img_name in list(bb[:,0]):
    shutil.copyfile(img_name,'imgs/flower_sub1000')
    assert(False)
