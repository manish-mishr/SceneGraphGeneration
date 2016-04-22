#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
To extarct CNN features.

'''


import os
os.environ["CHAINER_TYPE_CHECK"] = "0" #to disable type check
import chainer 

import argparse
import os
import numpy as np
from chainer import cuda
import chainer.functions as F
from chainer.functions import caffe
from chainer import cuda, Function, FunctionSet, gradient_check, Variable, optimizers
#import matplotlib.pyplot as plt
from scipy.misc import imread, imresize, imsave
import json
import nltk
import random
import pickle
import cPickle as cpickle
import math
import skimage.transform
import unicodecsv as csv
from tqdm import tqdm
import cv2

#Settings can be changed by command line arguments
gpu_id=0# GPU ID. if you want to use cpu, -1
#gpu_id=0

#Gpu Setting
if gpu_id >= 0:
    xp = cuda.cupy 
    cuda.get_device(gpu_id).use()
else:
    xp=np


MEAN_VALUES = np.array([104, 117, 123]).reshape((3,1,1))


#Load Caffe Model
cnn_model_place="../chainer_caption_generation/data/bvlc_googlenet_caffe_chainer.pkl"
with open(cnn_model_place, 'r') as f:
    func = pickle.load(f)
if gpu_id>= 0:
    func.to_gpu()
print "done"

def feature_dic_extractor(image_ids):
    image_id2feature={}
    for image_id in tqdm(image_ids):

        img_file="../data/images/%d.jpg"%image_id

        try:
            img=cv2.imread(img_file)
            img= cv2.resize(img,(224,224))
            img=img.transpose(2, 0, 1)-MEAN_VALUES
            img=img.astype(xp.float32)
        except Exception as e:
            print 'image reading error'
            print 'type:' + str(type(e))
            print 'args:' + str(e.args)
            print 'message:' + e.message
            print image_id
            continue

        x_batch = np.ndarray((1, 3, 224,224), dtype=np.float32)
        x_batch[0]=img
        if gpu_id >=0:
            x = Variable(cuda.to_gpu(x_batch), volatile=True)
        else:
            x = Variable(x_batch, volatile=True)
        image_feature_chainer, = func(inputs={'data': x}, outputs=['pool5/7x7_s1'],
                      disable=['loss1/ave_pool', 'loss2/ave_pool','loss3/classifier'],
                      train=False)

        image_feature_np=image_feature_chainer.data.reshape(1024)
        image_id2feature[image_id]=cuda.to_cpu(image_feature_np)

    return image_id2feature

with open('../work/relationship_train_image_ids.csv', 'r') as f:
    reader = csv.reader(f)
    relationship_train_image_ids=reader.next()
    relationship_train_image_ids = map(int,relationship_train_image_ids)
    
with open('../work/relationship_val_image_ids.csv', 'r') as f:
    reader = csv.reader(f)
    relationship_val_image_ids=reader.next()
    relationship_val_image_ids = map(int,relationship_val_image_ids)

with open('../work/relationship_test_image_ids.csv', 'r') as f:
    reader = csv.reader(f)
    relationship_test_image_ids=reader.next()
    relationship_test_image_ids = map(int,relationship_test_image_ids)

with open('../work/attribute_train_image_ids.csv', 'r') as f:
    reader = csv.reader(f)
    attribute_train_image_ids=reader.next()
    attribute_train_image_ids = map(int,attribute_train_image_ids)
    
with open('../work/attribute_val_image_ids.csv', 'r') as f:
    reader = csv.reader(f)
    attribute_val_image_ids=reader.next()
    attribute_val_image_ids = map(int,attribute_val_image_ids)

with open('../work/attribute_test_image_ids.csv', 'r') as f:
    reader = csv.reader(f)
    attribute_test_image_ids=reader.next()
    attribute_test_image_ids = map(int,attribute_test_image_ids)

attribute_train_features=feature_dic_extractor(attribute_train_image_ids)
attribute_val_features=feature_dic_extractor(attribute_val_image_ids)
attribute_test_features=feature_dic_extractor(attribute_test_image_ids)

with open('../work/attribute_train_features.cpickle', 'w') as f:
    cpickle.dump(attribute_train_features,f)
with open('../work/attribute_val_features.cpickle', 'w') as f:
    cpickle.dump(attribute_val_features,f)
with open('../work/attribute_test_features.cpickle', 'w') as f:
    cpickle.dump(attribute_test_features,f)

