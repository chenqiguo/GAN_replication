"""
Implementation of 'Maximum Likelihood Estimation of Intrinsic Dimension' by Elizaveta Levina and Peter J. Bickel

how to use
----------

The goal is to estimate intrinsic dimensionality of data, the estimation of dimensionality is scale dependent
(depending on how much you zoom into the data distribution you can find different dimesionality), so they
propose to average it over different scales, the interval of the scales [k1, k2] are the only parameters of the algorithm.

This code also provides a way to repeat the estimation with bootstrapping to estimate uncertainty.
"""


# This is my test code:
# my 1st try to compute intrinsic dimension on different image dataset.
# Note: this code is for GAN project!

# referenced from the original code: intdim_mle.py.



from tqdm import tqdm
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt

import cv2
import os
import pickle

from sklearn.datasets import make_swiss_roll # just for the example


"""
#### for FLOWER_128_sub1000: 1000 images dataset
srcRootDir_originDataImg = '/eecf/cbcsl/data100b/Chenqi/stylegan2/datasets_images/FLOWER_128_sub1000/jpg/' # these images are generated from tfrecords using code mycode_loadImgFromTFrecords.py
dstRootDir_figName = '/eecf/cbcsl/data100b/Chenqi/dataset_complexity/hist_intdim_mle/FLOWER_128_sub1000/'
dstRootDir_pkl = '/eecf/cbcsl/data100b/Chenqi/dataset_complexity/hist_intdim_mle/FLOWER_128_sub1000/intdim_k_repeated_dicts.pkl'
"""

#### for FLOWER_128_sub4000: 4000 images dataset
srcRootDir_originDataImg = '/eecf/cbcsl/data100b/Chenqi/stylegan2/datasets_images/FLOWER_128_sub4000/jpg/' # these images are generated from tfrecords using code mycode_loadImgFromTFrecords.py
dstRootDir_figName = '/eecf/cbcsl/data100b/Chenqi/dataset_complexity/hist_intdim_mle/FLOWER_128_sub4000/'
dstRootDir_pkl = '/eecf/cbcsl/data100b/Chenqi/dataset_complexity/hist_intdim_mle/FLOWER_128_sub4000/intdim_k_repeated_dicts.pkl'



biFlag = False


def intrinsic_dim_sample_wise(X, k=5):
    neighb = NearestNeighbors(n_neighbors=k + 1).fit(X)
    dist, ind = neighb.kneighbors(X)
    dist = dist[:, 1:]
    dist = dist[:, 0:k]
    assert dist.shape == (X.shape[0], k)
    assert np.all(dist > 0)
    d = np.log(dist[:, k - 1: k] / dist[:, 0:k-1])
    d = d.sum(axis=1) / (k - 2)
    d = 1. / d
    intdim_sample = d
    return intdim_sample

def intrinsic_dim_scale_interval(X, k1=10, k2=20):
    X = pd.DataFrame(X).drop_duplicates().values # remove duplicates in case you use bootstrapping
    intdim_k = []
    for k in range(k1, k2 + 1):
        m = intrinsic_dim_sample_wise(X, k).mean()
        intdim_k.append(m)
    return intdim_k

def repeated(func, X, nb_iter=100, random_state=None, verbose=0, mode='bootstrap', **func_kw):
    if random_state is None:
        rng = np.random
    else:
        rng = np.random.RandomState(random_state)
    nb_examples = X.shape[0]
    results = []

    iters = range(nb_iter)
    if verbose > 0:
        iters = tqdm(iters)    
    for i in iters:
        if mode == 'bootstrap':
            Xr = X[rng.randint(0, nb_examples, size=nb_examples)]
        elif mode == 'shuffle':
            ind = np.arange(nb_examples)
            rng.shuffle(ind)
            Xr = X[ind]
        elif mode == 'same':
            Xr = X
        else:
            raise ValueError('unknown mode : {}'.format(mode))
        results.append(func(Xr, **func_kw))
    return results


def image_to_feature_vector(image):
    # Note: the image is already resized to a fixed size.
	# flatten the image into a list of raw pixel intensities:
    
	return image.flatten()


def generateData(len_featVec, dim):
    # referenced from func generateTrainSet()
    # generate data X from image dataset (each row represents an image)
    all_origin_img_vecs = []
    for (dirpath, dirnames, filenames) in os.walk(srcRootDir_originDataImg):
        for filename in filenames:
            if ".jpg" in filename:
                #print("------------------deal with---------------------")
                #print(filename)
                origin_img = cv2.imread(srcRootDir_originDataImg+filename)
                if biFlag:
                    origin_img = origin_img[:,:,0]
                """
                # NO need to do this here: already 128x128 !
                origin_img_centCrop = my_center_crop(origin_img, min(origin_img.shape[0],origin_img.shape[1]))
                # resize using linear interpolation:
                origin_img_centCrop_resize = cv2.resize(origin_img_centCrop, dim)
                """
                # convert it to feature vector:
                origin_img_centCrop_resize_vec = image_to_feature_vector(origin_img)
                assert(len(origin_img_centCrop_resize_vec)==len_featVec)
                all_origin_img_vecs.append(origin_img_centCrop_resize_vec)
                
    return np.array(all_origin_img_vecs)
    


if __name__ == '__main__':
    """
    # the example:
    X, _ = make_swiss_roll(1000)
    k1 = 10 # start of interval(included)
    k2 = 20 # end of interval(included)
    intdim_k_repeated = repeated(intrinsic_dim_scale_interval, 
                                 X, 
                                 mode='bootstrap', 
                                 nb_iter=500, # nb_iter for bootstrapping
                                 verbose=1, 
                                 k1=k1, k2=k2)
    intdim_k_repeated = np.array(intdim_k_repeated)
    # the shape of intdim_k_repeated is (nb_iter, size_of_interval) where 
    # nb_iter is number of bootstrap iterations (here 500) and size_of_interval
    # is (k2 - k1 + 1).
    
    # Plotting the histogram of intrinsic dimensionality estimations repeated over
    # nb_iter experiments
    plt.hist(intdim_k_repeated.mean(axis=1))
    """
    
    # our test: on dataset FLOWER_128_sub4000:
    len_featVec = 128*128*3
    dim = (128, 128)
    X = generateData(len_featVec, dim)
    # grid search on below hyper-params!!!:
    k1_list = [10] # start of interval(included)
    k2_list = [20] # end of interval(included)
    nb_iter = 100
    for i in range(1):
        k1 = k1_list[i]
        k2 = k2_list[i]
        intdim_k_repeated = repeated(intrinsic_dim_scale_interval,
                                     X, 
                                     mode='bootstrap', 
                                     nb_iter=nb_iter, # nb_iter for bootstrapping
                                     verbose=1, 
                                     k1=k1, k2=k2)
        intdim_k_repeated = np.array(intdim_k_repeated)
        # the shape of intdim_k_repeated is (nb_iter, size_of_interval) where 
        # nb_iter is number of bootstrap iterations (here 500) and size_of_interval
        # is (k2 - k1 + 1).
        
        # Plotting the histogram of intrinsic dimensionality estimations repeated over
        # nb_iter experiments
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.hist(intdim_k_repeated.mean(axis=1))
        title_str = 'FLOWER_128_sub4000' + ' k1=' + str(k1) +' k2=' + str(k2) + ' nb_iter=' + str(nb_iter) + ' bootstrap'
        plt.title(title_str)
        fig.savefig(dstRootDir_figName + title_str + '.png')
        

    
    # run each time to save into pkl:
    all_store_dicts = []
    store_dict = {'dataset': 'FLOWER_128_sub4000',
                  'from_GAN': 'stylegan2',
                  'k1': k1,
                  'k2': k2,
                  'nb_iter': nb_iter,
                  'mode': 'bootstrap',
                  'intdim_k_repeated': intdim_k_repeated}
    all_store_dicts.append(store_dict)
    f_pkl = open(dstRootDir_pkl, 'wb')
    pickle.dump(all_store_dicts, f_pkl)
    f_pkl.close()
    
    """
    # our test: on dataset FLOWER_128_sub1000:
    len_featVec = 128*128*3
    dim = (128, 128)
    X = generateData(len_featVec, dim)
    # grid search on below hyper-params!!!:
    k1_list = [10,100,1000,10000] # start of interval(included)
    k2_list = [20,200,2000,15000] # end of interval(included)
    nb_iter = 200
    for i in range(4):
        k1 = k1_list[i]
        k2 = k2_list[i]
        intdim_k_repeated = repeated(intrinsic_dim_scale_interval,
                                     X, 
                                     mode='bootstrap', 
                                     nb_iter=nb_iter, # nb_iter for bootstrapping
                                     verbose=1, 
                                     k1=k1, k2=k2)
        intdim_k_repeated = np.array(intdim_k_repeated)
        # the shape of intdim_k_repeated is (nb_iter, size_of_interval) where 
        # nb_iter is number of bootstrap iterations (here 500) and size_of_interval
        # is (k2 - k1 + 1).
        
        # Plotting the histogram of intrinsic dimensionality estimations repeated over
        # nb_iter experiments
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.hist(intdim_k_repeated.mean(axis=1))
        title_str = 'FLOWER_128_sub1000' + ' k1=' + str(k1) +' k2=' + str(k2) + ' nb_iter=' + str(nb_iter) + ' bootstrap'
        plt.title(title_str)
        fig.savefig(dstRootDir_figName + title_str + '.png')
        

    
    # run each time to save into pkl:
    all_store_dicts = []
    store_dict = {'dataset': 'FLOWER_128_sub1000',
                  'from_GAN': 'stylegan2',
                  'k1': k1,
                  'k2': k2,
                  'nb_iter': nb_iter,
                  'mode': 'bootstrap',
                  'intdim_k_repeated': intdim_k_repeated}
    all_store_dicts.append(store_dict)
    f_pkl = open(dstRootDir_pkl, 'wb')
    pickle.dump(all_store_dicts, f_pkl)
    f_pkl.close()
    """



