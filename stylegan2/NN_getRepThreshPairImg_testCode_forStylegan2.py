#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 26 21:48:12 2020

@author: guo.1648
"""

# referenced from NN_getRepThreshPairImg_testCode_forBiggan.py.

# put the NN matching pair images with L2-norm distance smaller than threshold = 10000
# into the dst folder.

import re
import numpy as np
from shutil import copyfile

#5000 #6000 #7000 #8000 #9000 #10000 # <-- for pixel-wise matching
#0.35 #0.4 #0.45 #0.5 # <-- for simCLR matching
#10 11 12 13 14 15 <- for inception v3
NNmatchDist_threshold_value = 10000
total_sample_num = 1024

"""
#### tmp: inception v3:
# for biggan FLOWER_128_sub1000: 1000 images dataset 
# for Itr38950:
srcRootDir_img = '/eecf/cbcsl/data100b/Chenqi/gan_results_for_presentation/biggan/NNquery_inception_v3/FLOWER_128_sub1000/Itr38950/NNmatchResult/'
srcTxtFile = '/eecf/cbcsl/data100b/Chenqi/gan_results_for_presentation/biggan/NNquery_inception_v3/FLOWER_128_sub1000/Itr38950/NNmatchDist.txt'

dstRootDir_img = '/eecf/cbcsl/data100b/Chenqi/gan_results_for_presentation/biggan/NNquery_inception_v3/FLOWER_128_sub1000/Itr38950/NNmatchResult_threshold15/'
dstTxtName_matchDistThresh = '/eecf/cbcsl/data100b/Chenqi/gan_results_for_presentation/biggan/NNquery_inception_v3/FLOWER_128_sub1000/Itr38950/NNmatchDist_smallerThanThresh.txt'
"""
"""
#### tmp: inception v3:
# for stylegan2 FLOWER_128_sub1000: 1000 images dataset 
# for fakes003248:
srcRootDir_img = '/eecf/cbcsl/data100b/Chenqi/gan_results_for_presentation/stylegan2/NNquery_inception_v3/FLOWER_128_sub1000_resume/fakes003248/NNmatchResult/'
srcTxtFile = '/eecf/cbcsl/data100b/Chenqi/gan_results_for_presentation/stylegan2/NNquery_inception_v3/FLOWER_128_sub1000_resume/fakes003248/NNmatchDist.txt'

dstRootDir_img = '/eecf/cbcsl/data100b/Chenqi/gan_results_for_presentation/stylegan2/NNquery_inception_v3/FLOWER_128_sub1000_resume/fakes003248/NNmatchResult_threshold15/'
dstTxtName_matchDistThresh = '/eecf/cbcsl/data100b/Chenqi/gan_results_for_presentation/stylegan2/NNquery_inception_v3/FLOWER_128_sub1000_resume/fakes003248/NNmatchDist_smallerThanThresh.txt'
"""



"""
#### for FLOWER_128: 8189 images dataset (the original FLOWER dataset)
# for fakes002526:
srcRootDir_img = '/eecf/cbcsl/data100b/Chenqi/stylegan2/imgs/NN_query/FLOWER_128/fakes002526/NNmatchResult/'
srcTxtFile = '/eecf/cbcsl/data100b/Chenqi/stylegan2/imgs/NN_query/FLOWER_128/fakes002526/NNmatchDist.txt'

dstRootDir_img = '/eecf/cbcsl/data100b/Chenqi/gan_results_for_presentation/stylegan2/NN_query/FLOWER_128/fakes002526/NNmatchResult_threshold7000/'
dstTxtName_matchDistThresh = '/eecf/cbcsl/data100b/Chenqi/stylegan2/imgs/NN_query/FLOWER_128/fakes002526/NNmatchDist_smallerThanThresh.txt'
"""
"""
#### for simCLR FLOWER_128: 8189 images dataset (the original FLOWER dataset)
# for fakes002526:
srcRootDir_img = '/eecf/cbcsl/data100b/Chenqi/stylegan2/imgs/NNquery_simCLR_v2/FLOWER_128/fakes002526/NNmatchResult/'
srcTxtFile = '/eecf/cbcsl/data100b/Chenqi/stylegan2/imgs/NNquery_simCLR_v2/FLOWER_128/fakes002526/NNmatchDist.txt'

dstRootDir_img = '/eecf/cbcsl/data100b/Chenqi/gan_results_for_presentation/stylegan2/NNquery_simCLR_v2/FLOWER_128/fakes002526/NNmatchResult_threshold0.4/'
dstTxtName_matchDistThresh = '/eecf/cbcsl/data100b/Chenqi/gan_results_for_presentation/stylegan2/NNquery_simCLR_v2/FLOWER_128/fakes002526/NNmatchDist_smallerThanThresh.txt'
"""
"""
# for rebuttal:
#### for FLOWER_128: 8189 images dataset
# for fakes001925:
srcRootDir_img = '/eecf/cbcsl/data100b/Chenqi/stylegan2/imgs/NN_query/FLOWER_128/fakes001925/NNmatchResult/'
srcTxtFile = '/eecf/cbcsl/data100b/Chenqi/stylegan2/imgs/NN_query/FLOWER_128/fakes001925/NNmatchDist.txt'

dstRootDir_img = '/eecf/cbcsl/data100b/Chenqi/gan_results_for_presentation/stylegan2/NN_query/FLOWER_128/fakes001925/NNmatchResult_threshold10000/'
dstTxtName_matchDistThresh = '/eecf/cbcsl/data100b/Chenqi/gan_results_for_presentation/stylegan2/NN_query/FLOWER_128/fakes001925/NNmatchDist_smallerThanThresh.txt'
"""

"""
#### for FLOWER_128_sub1000: 1000 images dataset (resume)
# for fakes003248:
srcRootDir_img = '/eecf/cbcsl/data100b/Chenqi/stylegan2/imgs/NN_query/FLOWER_128_sub1000_resume/fakes003248/NNmatchResult/'
srcTxtFile = '/eecf/cbcsl/data100b/Chenqi/stylegan2/imgs/NN_query/FLOWER_128_sub1000_resume/fakes003248/NNmatchDist.txt'

dstRootDir_img = '/eecf/cbcsl/data100b/Chenqi/gan_results_for_presentation/stylegan2/NN_query/FLOWER_128_sub1000_resume/fakes003248/NNmatchResult_threshold7000/'
dstTxtName_matchDistThresh = '/eecf/cbcsl/data100b/Chenqi/stylegan2/imgs/NN_query/FLOWER_128_sub1000_resume/fakes003248/NNmatchDist_smallerThanThresh.txt'
"""
"""
#### for simCLR FLOWER_128_sub1000: 1000 images dataset (resume)
# for fakes003248:
srcRootDir_img = '/eecf/cbcsl/data100b/Chenqi/stylegan2/imgs/NNquery_simCLR_v2/FLOWER_128_sub1000_resume/fakes003248/NNmatchResult/'
srcTxtFile = '/eecf/cbcsl/data100b/Chenqi/stylegan2/imgs/NNquery_simCLR_v2/FLOWER_128_sub1000_resume/fakes003248/NNmatchDist.txt'

dstRootDir_img = '/eecf/cbcsl/data100b/Chenqi/gan_results_for_presentation/stylegan2/NNquery_simCLR_v2/FLOWER_128_sub1000_resume/fakes003248/NNmatchResult_threshold0.4/'
dstTxtName_matchDistThresh = '/eecf/cbcsl/data100b/Chenqi/gan_results_for_presentation/stylegan2/NNquery_simCLR_v2/FLOWER_128_sub1000_resume/fakes003248/NNmatchDist_smallerThanThresh.txt'
"""
"""
# for rebuttal:
#### for FLOWER_128_sub1000: 1000 images dataset
# for fakes001684:
srcRootDir_img = '/eecf/cbcsl/data100b/Chenqi/stylegan2/imgs/NN_query/FLOWER_128_sub1000/fakes001684/NNmatchResult/'
srcTxtFile = '/eecf/cbcsl/data100b/Chenqi/stylegan2/imgs/NN_query/FLOWER_128_sub1000/fakes001684/NNmatchDist.txt'

dstRootDir_img = '/eecf/cbcsl/data100b/Chenqi/gan_results_for_presentation/stylegan2/NN_query/FLOWER_128_sub1000/fakes001684/NNmatchResult_threshold10000/'
dstTxtName_matchDistThresh = '/eecf/cbcsl/data100b/Chenqi/gan_results_for_presentation/stylegan2/NN_query/FLOWER_128_sub1000/fakes001684/NNmatchDist_smallerThanThresh.txt'
"""


"""
#### for FLOWER_128_sub4000: 4000 images dataset (resume)
# for fakes003248:
srcRootDir_img = '/eecf/cbcsl/data100b/Chenqi/stylegan2/imgs/NN_query/FLOWER_128_sub4000_resume/fakes003248/NNmatchResult/'
srcTxtFile = '/eecf/cbcsl/data100b/Chenqi/stylegan2/imgs/NN_query/FLOWER_128_sub4000_resume/fakes003248/NNmatchDist.txt'

dstRootDir_img = '/eecf/cbcsl/data100b/Chenqi/gan_results_for_presentation/stylegan2/NN_query/FLOWER_128_sub4000_resume/fakes003248/NNmatchResult_threshold7000/'
dstTxtName_matchDistThresh = '/eecf/cbcsl/data100b/Chenqi/stylegan2/imgs/NN_query/FLOWER_128_sub4000_resume/fakes003248/NNmatchDist_smallerThanThresh.txt'
"""
"""
#### for simCLR FLOWER_128_sub4000: 4000 images dataset (resume)
# for fakes003248:
srcRootDir_img = '/eecf/cbcsl/data100b/Chenqi/stylegan2/imgs/NNquery_simCLR_v2/FLOWER_128_sub4000_resume/fakes003248/NNmatchResult/'
srcTxtFile = '/eecf/cbcsl/data100b/Chenqi/stylegan2/imgs/NNquery_simCLR_v2/FLOWER_128_sub4000_resume/fakes003248/NNmatchDist.txt'

dstRootDir_img = '/eecf/cbcsl/data100b/Chenqi/gan_results_for_presentation/stylegan2/NNquery_simCLR_v2/FLOWER_128_sub4000_resume/fakes003248/NNmatchResult_threshold0.35/'
dstTxtName_matchDistThresh = '/eecf/cbcsl/data100b/Chenqi/gan_results_for_presentation/stylegan2/NNquery_simCLR_v2/FLOWER_128_sub4000_resume/fakes003248/NNmatchDist_smallerThanThresh.txt'
"""
"""
# for rebuttal:
#### for FLOWER_128_sub4000: 4000 images dataset
# for fakes001925:
srcRootDir_img = '/eecf/cbcsl/data100b/Chenqi/stylegan2/imgs/NN_query/FLOWER_128_sub4000/fakes001925/NNmatchResult/'
srcTxtFile = '/eecf/cbcsl/data100b/Chenqi/stylegan2/imgs/NN_query/FLOWER_128_sub4000/fakes001925/NNmatchDist.txt'

dstRootDir_img = '/eecf/cbcsl/data100b/Chenqi/gan_results_for_presentation/stylegan2/NN_query/FLOWER_128_sub4000/fakes001925/NNmatchResult_threshold10000/'
dstTxtName_matchDistThresh = '/eecf/cbcsl/data100b/Chenqi/gan_results_for_presentation/stylegan2/NN_query/FLOWER_128_sub4000/fakes001925/NNmatchDist_smallerThanThresh.txt'
"""

"""
#### for CelebA_128_sub200: 200 images dataset
# for fakes007700:
srcRootDir_img = '/eecf/cbcsl/data100b/Chenqi/stylegan2/imgs/NN_query/CelebA_128_sub200/fakes007700/NNmatchResult/'
srcTxtFile = '/eecf/cbcsl/data100b/Chenqi/stylegan2/imgs/NN_query/CelebA_128_sub200/fakes007700/NNmatchDist.txt'

dstRootDir_img = '/eecf/cbcsl/data100b/Chenqi/gan_results_for_presentation/stylegan2/NN_query/CelebA_128_sub200/fakes007700/NNmatchResult_threshold7000/'
dstTxtName_matchDistThresh = '/eecf/cbcsl/data100b/Chenqi/gan_results_for_presentation/stylegan2/NN_query/CelebA_128_sub200/fakes007700/NNmatchDist_smallerThanThresh.txt'
"""
"""
#### for CelebA_128_sub600: 600 images dataset
# for fakes005414:
srcRootDir_img = '/eecf/cbcsl/data100b/Chenqi/stylegan2/imgs/NN_query/CelebA_128_sub600/fakes005414/NNmatchResult/'
srcTxtFile = '/eecf/cbcsl/data100b/Chenqi/stylegan2/imgs/NN_query/CelebA_128_sub600/fakes005414/NNmatchDist.txt'

dstRootDir_img = '/eecf/cbcsl/data100b/Chenqi/gan_results_for_presentation/stylegan2/NN_query/CelebA_128_sub600/fakes005414/NNmatchResult_threshold10000/'
dstTxtName_matchDistThresh = '/eecf/cbcsl/data100b/Chenqi/gan_results_for_presentation/stylegan2/NN_query/CelebA_128_sub600/fakes005414/NNmatchDist_smallerThanThresh.txt'
"""
"""
#### for CelebA_128_sub1000: 1000 images dataset
# for fakes004933:
srcRootDir_img = '/eecf/cbcsl/data100b/Chenqi/stylegan2/imgs/NN_query/CelebA_128_sub1000/fakes004933/NNmatchResult/'
srcTxtFile = '/eecf/cbcsl/data100b/Chenqi/stylegan2/imgs/NN_query/CelebA_128_sub1000/fakes004933/NNmatchDist.txt'

dstRootDir_img = '/eecf/cbcsl/data100b/Chenqi/gan_results_for_presentation/stylegan2/NN_query/CelebA_128_sub1000/fakes004933/NNmatchResult_threshold10000/'
dstTxtName_matchDistThresh = '/eecf/cbcsl/data100b/Chenqi/gan_results_for_presentation/stylegan2/NN_query/CelebA_128_sub1000/fakes004933/NNmatchDist_smallerThanThresh.txt'
"""
"""
#### for CelebA_128_sub4000: 4000 images dataset
# for fakes003369:
srcRootDir_img = '/eecf/cbcsl/data100b/Chenqi/stylegan2/imgs/NN_query/CelebA_128_sub4000/fakes003369/NNmatchResult/'
srcTxtFile = '/eecf/cbcsl/data100b/Chenqi/stylegan2/imgs/NN_query/CelebA_128_sub4000/fakes003369/NNmatchDist.txt'

dstRootDir_img = '/eecf/cbcsl/data100b/Chenqi/gan_results_for_presentation/stylegan2/NN_query/CelebA_128_sub4000/fakes003369/NNmatchResult_threshold10000/'
dstTxtName_matchDistThresh = '/eecf/cbcsl/data100b/Chenqi/gan_results_for_presentation/stylegan2/NN_query/CelebA_128_sub4000/fakes003369/NNmatchDist_smallerThanThresh.txt'
"""
"""
#### for CelebA_128_sub8000: 8000 images dataset
# for fakes001684:
srcRootDir_img = '/eecf/cbcsl/data100b/Chenqi/stylegan2/imgs/NN_query/CelebA_128_sub8000/fakes001684/NNmatchResult/'
srcTxtFile = '/eecf/cbcsl/data100b/Chenqi/stylegan2/imgs/NN_query/CelebA_128_sub8000/fakes001684/NNmatchDist.txt'

dstRootDir_img = '/eecf/cbcsl/data100b/Chenqi/gan_results_for_presentation/stylegan2/NN_query/CelebA_128_sub8000/fakes001684/NNmatchResult_threshold10000/'
dstTxtName_matchDistThresh = '/eecf/cbcsl/data100b/Chenqi/gan_results_for_presentation/stylegan2/NN_query/CelebA_128_sub8000/fakes001684/NNmatchDist_smallerThanThresh.txt'
"""
"""
#### for MNIST_128_sub10000: 10000 images dataset
# for fakes005173:
srcRootDir_img = '/eecf/cbcsl/data100b/Chenqi/stylegan2/imgs/NN_query/MNIST_128_sub10000/fakes005173/NNmatchResult/'
srcTxtFile = '/eecf/cbcsl/data100b/Chenqi/stylegan2/imgs/NN_query/MNIST_128_sub10000/fakes005173/NNmatchDist.txt'

dstRootDir_img = '/eecf/cbcsl/data100b/Chenqi/gan_results_for_presentation/stylegan2/NN_query/MNIST_128_sub10000/fakes005173/NNmatchResult_threshold7000/'
dstTxtName_matchDistThresh = '/eecf/cbcsl/data100b/Chenqi/gan_results_for_presentation/stylegan2/NN_query/MNIST_128_sub10000/fakes005173/NNmatchDist_smallerThanThresh.txt'
"""
"""
#### for MNIST_128_sub10000: 10000 images dataset, 3ch:
# for fakes005173:
srcRootDir_img = '/eecf/cbcsl/data100b/Chenqi/stylegan2/imgs/NN_query/MNIST_128_sub10000_3ch/fakes005173/NNmatchResult/'
srcTxtFile = '/eecf/cbcsl/data100b/Chenqi/stylegan2/imgs/NN_query/MNIST_128_sub10000_3ch/fakes005173/NNmatchDist.txt'

dstRootDir_img = '/eecf/cbcsl/data100b/Chenqi/gan_results_for_presentation/stylegan2/NN_query/MNIST_128_sub10000_3ch/fakes005173/NNmatchResult_threshold9000/'
dstTxtName_matchDistThresh = '/eecf/cbcsl/data100b/Chenqi/gan_results_for_presentation/stylegan2/NN_query/MNIST_128_sub10000_3ch/fakes005173/NNmatchDist_smallerThanThresh.txt'
"""
"""
#### for MNIST_128_sub30000: 30000 images dataset
# for fakes005053:
srcRootDir_img = '/eecf/cbcsl/data100b/Chenqi/stylegan2/imgs/NN_query/MNIST_128_sub30000/fakes005053/NNmatchResult/'
srcTxtFile = '/eecf/cbcsl/data100b/Chenqi/stylegan2/imgs/NN_query/MNIST_128_sub30000/fakes005053/NNmatchDist.txt'

dstRootDir_img = '/eecf/cbcsl/data100b/Chenqi/gan_results_for_presentation/stylegan2/NN_query/MNIST_128_sub30000/fakes005053/NNmatchResult_threshold7000/'
dstTxtName_matchDistThresh = '/eecf/cbcsl/data100b/Chenqi/gan_results_for_presentation/stylegan2/NN_query/MNIST_128_sub30000/fakes005053/NNmatchDist_smallerThanThresh.txt'
"""
"""
#### for MNIST_128_sub30000: 30000 images dataset, bi:
# for fakes005053:
srcRootDir_img = '/eecf/cbcsl/data100b/Chenqi/stylegan2/imgs/NN_query/MNIST_128_sub30000_bi/fakes005053/NNmatchResult/'
srcTxtFile = '/eecf/cbcsl/data100b/Chenqi/stylegan2/imgs/NN_query/MNIST_128_sub30000_bi/fakes005053/NNmatchDist.txt'

dstRootDir_img = '/eecf/cbcsl/data100b/Chenqi/gan_results_for_presentation/stylegan2/NN_query/MNIST_128_sub30000_bi/fakes005053/NNmatchResult_threshold8000/'
dstTxtName_matchDistThresh = '/eecf/cbcsl/data100b/Chenqi/gan_results_for_presentation/stylegan2/NN_query/MNIST_128_sub30000_bi/fakes005053/NNmatchDist_smallerThanThresh.txt'
"""
"""
#### for MNIST_128_train: 60000 images dataset
# for fakes003609:
srcRootDir_img = '/eecf/cbcsl/data100b/Chenqi/stylegan2/imgs/NN_query/MNIST_128_train/fakes003609/NNmatchResult/'
srcTxtFile = '/eecf/cbcsl/data100b/Chenqi/stylegan2/imgs/NN_query/MNIST_128_train/fakes003609/NNmatchDist.txt'

dstRootDir_img = '/eecf/cbcsl/data100b/Chenqi/gan_results_for_presentation/stylegan2/NN_query/MNIST_128_train/fakes003609/NNmatchResult_threshold7000/'
dstTxtName_matchDistThresh = '/eecf/cbcsl/data100b/Chenqi/gan_results_for_presentation/stylegan2/NN_query/MNIST_128_train/fakes003609/NNmatchDist_smallerThanThresh.txt'
"""
"""
#### for LSUN_128_sub10000: 10000 images dataset, bi:
# for fakes004812:
srcRootDir_img = '/eecf/cbcsl/data100b/Chenqi/stylegan2/imgs/NN_query/LSUN_128_sub10000/fakes004812/NNmatchResult/'
srcTxtFile = '/eecf/cbcsl/data100b/Chenqi/stylegan2/imgs/NN_query/LSUN_128_sub10000/fakes004812/NNmatchDist.txt'

dstRootDir_img = '/eecf/cbcsl/data100b/Chenqi/gan_results_for_presentation/stylegan2/NN_query/LSUN_128_sub10000/fakes004812/NNmatchResult_threshold7000/'
dstTxtName_matchDistThresh = '/eecf/cbcsl/data100b/Chenqi/gan_results_for_presentation/stylegan2/NN_query/LSUN_128_sub10000/fakes004812/NNmatchDist_smallerThanThresh.txt'
"""
"""
#### for LSUN_128_sub30000: 30000 images dataset, bi:
# for fakes004692:
srcRootDir_img = '/eecf/cbcsl/data100b/Chenqi/stylegan2/imgs/NN_query/LSUN_128_sub30000/fakes004692/NNmatchResult/'
srcTxtFile = '/eecf/cbcsl/data100b/Chenqi/stylegan2/imgs/NN_query/LSUN_128_sub30000/fakes004692/NNmatchDist.txt'

dstRootDir_img = '/eecf/cbcsl/data100b/Chenqi/gan_results_for_presentation/stylegan2/NN_query/LSUN_128_sub30000/fakes004692/NNmatchResult_threshold7000/'
dstTxtName_matchDistThresh = '/eecf/cbcsl/data100b/Chenqi/gan_results_for_presentation/stylegan2/NN_query/LSUN_128_sub30000/fakes004692/NNmatchDist_smallerThanThresh.txt'
"""
"""
#### for LSUN_128_sub60000: 60000 images dataset, bi:
# for fakes006497:
srcRootDir_img = '/eecf/cbcsl/data100b/Chenqi/stylegan2/imgs/NN_query/LSUN_128_sub60000/fakes006497/NNmatchResult/'
srcTxtFile = '/eecf/cbcsl/data100b/Chenqi/stylegan2/imgs/NN_query/LSUN_128_sub60000/fakes006497/NNmatchDist.txt'

dstRootDir_img = '/eecf/cbcsl/data100b/Chenqi/gan_results_for_presentation/stylegan2/NN_query/LSUN_128_sub60000/fakes006497/NNmatchResult_threshold10000/'
dstTxtName_matchDistThresh = '/eecf/cbcsl/data100b/Chenqi/gan_results_for_presentation/stylegan2/NN_query/LSUN_128_sub60000/fakes006497/NNmatchDist_smallerThanThresh.txt'
"""
"""
#### for LSUN_128_sub1000_resume: 1000 images dataset, bi:
# for fakes002165:
srcRootDir_img = '/eecf/cbcsl/data100b/Chenqi/stylegan2/imgs/NN_query/LSUN_128_sub1000_resume/fakes002165/NNmatchResult/'
srcTxtFile = '/eecf/cbcsl/data100b/Chenqi/stylegan2/imgs/NN_query/LSUN_128_sub1000_resume/fakes002165/NNmatchDist.txt'

dstRootDir_img = '/eecf/cbcsl/data100b/Chenqi/gan_results_for_presentation/stylegan2/NN_query/LSUN_128_sub1000_resume/fakes002165/NNmatchResult_threshold7000/'
dstTxtName_matchDistThresh = '/eecf/cbcsl/data100b/Chenqi/gan_results_for_presentation/stylegan2/NN_query/LSUN_128_sub1000_resume/fakes002165/NNmatchDist_smallerThanThresh.txt'
"""
"""
#### for LSUN_128_sub5000_resume: 5000 images dataset, bi:
# for fakes000000:
srcRootDir_img = '/eecf/cbcsl/data100b/Chenqi/stylegan2/imgs/NN_query/LSUN_128_sub5000_resume/fakes000000/NNmatchResult/'
srcTxtFile = '/eecf/cbcsl/data100b/Chenqi/stylegan2/imgs/NN_query/LSUN_128_sub5000_resume/fakes000000/NNmatchDist.txt'

dstRootDir_img = '/eecf/cbcsl/data100b/Chenqi/gan_results_for_presentation/stylegan2/NN_query/LSUN_128_sub5000_resume/fakes000000/NNmatchResult_threshold7000/'
dstTxtName_matchDistThresh = '/eecf/cbcsl/data100b/Chenqi/gan_results_for_presentation/stylegan2/NN_query/LSUN_128_sub5000_resume/fakes000000/NNmatchDist_smallerThanThresh.txt'
"""
"""
#### for LSUN_128_sub200: 200 images dataset, bi:
# for fakes006497:
srcRootDir_img = '/eecf/cbcsl/data100b/Chenqi/stylegan2/imgs/NN_query/LSUN_128_sub200/fakes006497/NNmatchResult/'
srcTxtFile = '/eecf/cbcsl/data100b/Chenqi/stylegan2/imgs/NN_query/LSUN_128_sub200/fakes006497/NNmatchDist.txt'

dstRootDir_img = '/eecf/cbcsl/data100b/Chenqi/gan_results_for_presentation/stylegan2/NN_query/LSUN_128_sub200/fakes006497/NNmatchResult_threshold7000/'
dstTxtName_matchDistThresh = '/eecf/cbcsl/data100b/Chenqi/gan_results_for_presentation/stylegan2/NN_query/LSUN_128_sub200/fakes006497/NNmatchDist_smallerThanThresh.txt'
"""


### for rebuttal: CIFAR10:
"""
#### for CIFAR10_32_sub1000: 1000 images dataset
# for fakes002813:
srcRootDir_img = '/eecf/cbcsl/data100b/Chenqi/stylegan2/imgs/NN_query/CIFAR10_32_sub1000/fakes002813/NNmatchResult/'
srcTxtFile = '/eecf/cbcsl/data100b/Chenqi/stylegan2/imgs/NN_query/CIFAR10_32_sub1000/fakes002813/NNmatchDist.txt'

dstRootDir_img = '/eecf/cbcsl/data100b/Chenqi/gan_results_for_presentation/stylegan2/NN_query/CIFAR10_32_sub1000/fakes002813/NNmatchResult_threshold1800/'
dstTxtName_matchDistThresh = '/eecf/cbcsl/data100b/Chenqi/gan_results_for_presentation/stylegan2/NN_query/CIFAR10_32_sub1000/fakes002813/NNmatchDist_smallerThanThresh.txt'
"""
"""
#### for CIFAR10_32_sub4000: 4000 images dataset
# for fakes003014:
srcRootDir_img = '/eecf/cbcsl/data100b/Chenqi/stylegan2/imgs/NN_query/CIFAR10_32_sub4000/fakes003014/NNmatchResult/'
srcTxtFile = '/eecf/cbcsl/data100b/Chenqi/stylegan2/imgs/NN_query/CIFAR10_32_sub4000/fakes003014/NNmatchDist.txt'

dstRootDir_img = '/eecf/cbcsl/data100b/Chenqi/gan_results_for_presentation/stylegan2/NN_query/CIFAR10_32_sub4000/fakes003014/NNmatchResult_threshold1800/'
dstTxtName_matchDistThresh = '/eecf/cbcsl/data100b/Chenqi/gan_results_for_presentation/stylegan2/NN_query/CIFAR10_32_sub4000/fakes003014/NNmatchDist_smallerThanThresh.txt'
"""
"""
#### for CIFAR10_32_sub8000: 8000 images dataset
# for fakes003014:
srcRootDir_img = '/eecf/cbcsl/data100b/Chenqi/stylegan2/imgs/NN_query/CIFAR10_32_sub8000/fakes003014/NNmatchResult/'
srcTxtFile = '/eecf/cbcsl/data100b/Chenqi/stylegan2/imgs/NN_query/CIFAR10_32_sub8000/fakes003014/NNmatchDist.txt'

dstRootDir_img = '/eecf/cbcsl/data100b/Chenqi/gan_results_for_presentation/stylegan2/NN_query/CIFAR10_32_sub8000/fakes003014/NNmatchResult_threshold1800/'
dstTxtName_matchDistThresh = '/eecf/cbcsl/data100b/Chenqi/gan_results_for_presentation/stylegan2/NN_query/CIFAR10_32_sub8000/fakes003014/NNmatchDist_smallerThanThresh.txt'
"""
"""
#### for CIFAR10_32_sub10000: 10000 images dataset
# for fakes002009:
srcRootDir_img = '/eecf/cbcsl/data100b/Chenqi/stylegan2/imgs/NN_query/CIFAR10_32_sub10000/fakes002009/NNmatchResult/'
srcTxtFile = '/eecf/cbcsl/data100b/Chenqi/stylegan2/imgs/NN_query/CIFAR10_32_sub10000/fakes002009/NNmatchDist.txt'

dstRootDir_img = '/eecf/cbcsl/data100b/Chenqi/gan_results_for_presentation/stylegan2/NN_query/CIFAR10_32_sub10000/fakes002009/NNmatchResult_threshold1800/'
dstTxtName_matchDistThresh = '/eecf/cbcsl/data100b/Chenqi/gan_results_for_presentation/stylegan2/NN_query/CIFAR10_32_sub10000/fakes002009/NNmatchDist_smallerThanThresh.txt'
"""


NNmatchDist_threshold_value = 18000 #19000 # 20000 21000 23000 25000
total_sample_num = 480
"""
#### for FLOWER_256_sub1000: 1000 images dataset
# for fakes002009:
srcRootDir_img = '/eecf/cbcsl/data100b/Chenqi/stylegan2/imgs/NN_query/FLOWER_256_sub1000/fakes004435/NNmatchResult/'
srcTxtFile = '/eecf/cbcsl/data100b/Chenqi/stylegan2/imgs/NN_query/FLOWER_256_sub1000/fakes004435/NNmatchDist.txt'

dstRootDir_img = '/eecf/cbcsl/data100b/Chenqi/gan_results_for_presentation/stylegan2/NN_query/FLOWER_256_sub1000/00002/fakes004435/NNmatchResult_threshold18000/'
dstTxtName_matchDistThresh = '/eecf/cbcsl/data100b/Chenqi/gan_results_for_presentation/stylegan2/NN_query/FLOWER_256_sub1000/00002/fakes004435/NNmatchDist_smallerThanThresh.txt'
"""
"""
#### for FLOWER_256_sub4000: 4000 images dataset
# for fakes000000:
srcRootDir_img = '/eecf/cbcsl/data100b/Chenqi/stylegan2/imgs/NN_query/FLOWER_256_sub4000/fakes006128/NNmatchResult/'
srcTxtFile = '/eecf/cbcsl/data100b/Chenqi/stylegan2/imgs/NN_query/FLOWER_256_sub4000/fakes006128/NNmatchDist.txt'

dstRootDir_img = '/eecf/cbcsl/data100b/Chenqi/gan_results_for_presentation/stylegan2/NN_query/FLOWER_256_sub4000/00002/fakes006128/NNmatchResult_threshold18000/'
dstTxtName_matchDistThresh = '/eecf/cbcsl/data100b/Chenqi/gan_results_for_presentation/stylegan2/NN_query/FLOWER_256_sub4000/00002/fakes006128/NNmatchDist_smallerThanThresh.txt'
"""
"""
#### for FLOWER_256_sub6000: 6000 images dataset
# for fakes000403:
srcRootDir_img = '/eecf/cbcsl/data100b/Chenqi/stylegan2/imgs/NN_query/FLOWER_256_sub6000/fakes006290/NNmatchResult/'
srcTxtFile = '/eecf/cbcsl/data100b/Chenqi/stylegan2/imgs/NN_query/FLOWER_256_sub6000/fakes006290/NNmatchDist.txt'

dstRootDir_img = '/eecf/cbcsl/data100b/Chenqi/gan_results_for_presentation/stylegan2/NN_query/FLOWER_256_sub6000/00002/fakes006290/NNmatchResult_threshold18000/'
dstTxtName_matchDistThresh = '/eecf/cbcsl/data100b/Chenqi/gan_results_for_presentation/stylegan2/NN_query/FLOWER_256_sub6000/00002/fakes006290/NNmatchDist_smallerThanThresh.txt'
"""
#"""
#### for FLOWER_256: 8189 images dataset
# for fakes000161:
srcRootDir_img = '/eecf/cbcsl/data100b/Chenqi/stylegan2/imgs/NN_query/FLOWER_256/fakes006209/NNmatchResult/'
srcTxtFile = '/eecf/cbcsl/data100b/Chenqi/stylegan2/imgs/NN_query/FLOWER_256/fakes006209/NNmatchDist.txt'

dstRootDir_img = '/eecf/cbcsl/data100b/Chenqi/gan_results_for_presentation/stylegan2/NN_query/FLOWER_256/00002/fakes006209/NNmatchResult_threshold18000/'
dstTxtName_matchDistThresh = '/eecf/cbcsl/data100b/Chenqi/gan_results_for_presentation/stylegan2/NN_query/FLOWER_256/00002/fakes006209/NNmatchDist_smallerThanThresh.txt'
#"""




if __name__ == '__main__':
    match_distance_thresh_strs = ''
    match_num = 0
    for line in open(srcTxtFile):
        pairImgName = line.split(': match_distance = ')[0]
        l2Dist = float(line.split(': match_distance = ')[-1])
        #print()
        if l2Dist <= NNmatchDist_threshold_value:
            srcImgName = srcRootDir_img + pairImgName
            dstImgName = dstRootDir_img + pairImgName
            copyfile(srcImgName, dstImgName)
            
            match_num += 1
            match_distance_thresh_strs += line
        
    
    match_percent = match_num / total_sample_num * 100
    match_distance_thresh_strs = 'match_num = ' + str(match_num) + '\n' \
                                 + 'total sample_num = ' + str(total_sample_num) + '\n' \
                                 + 'match_percent = ' + str(match_percent) + '%\n' \
                                 + match_distance_thresh_strs
    # newly added: save the values related to match_distance smaller than thresh into txt file:
    f = open(dstTxtName_matchDistThresh, 'w')
    f.write(match_distance_thresh_strs)
    f.close()

