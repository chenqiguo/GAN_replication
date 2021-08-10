#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 26 11:52:17 2021

@author: guo.1648
"""

f_str = ''
i = 1

with open('/eecf/cbcsl/data100b/Chenqi/data/myFile.txt') as file_in:
    for line in file_in:
        img_name = 'image_' + str(i).zfill(5) + '.jpg'
        f_str += img_name + ' ' + line
        i += 1

text_file = open("/eecf/cbcsl/data100b/Chenqi/data/flower_labels.txt", "w")
text_file.write(f_str)
text_file.close()


