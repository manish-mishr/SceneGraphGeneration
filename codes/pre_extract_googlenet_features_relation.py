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

def get_image_relationships(image_id,all_relationships):
    for i in xrange(len(all_relationships)):
        image_id_=all_relationships[i][u'id']
        if image_id==image_id_:
            relationships=all_relationships[i][u'relationships']
    return relationships

def pre_extract_relation_feature(image_ids):
    triple_id2subject_feature={}
    triple_id2object_feature={}
    triple_id2relation_id={}
    for image_id in tqdm(image_ids):
        image_relationships=get_image_relationships(image_id,all_relationships)
        img_file="../data/images/%d.jpg"%image_id
        img=cv2.imread(img_file,cv2.IMREAD_COLOR)
        all_img_list=[]
        triple_ids_tmp=[]
        for triple in image_relationships:
            triple_id=triple[u'id']
            triple_predicate=triple[u'predicate']

            if triple_predicate not in relation2index:
                continue
            relation_id=relation2index[triple_predicate]

            triple_subject=triple[u'subject']
            triple_subject_name=triple_subject['name']
            triple_subject_x=triple_subject['x']
            triple_subject_y=triple_subject['y']
            triple_subject_w=triple_subject['w']
            triple_subject_h=triple_subject['h']

            triple_object=triple[u'object']
            triple_object_name=triple_object['name']
            triple_object_x=triple_object['x']
            triple_object_y=triple_object['y']
            triple_object_w=triple_object['w']
            triple_object_h=triple_object['h']
            
            try:
                img_sbj=img[triple_subject_y:triple_subject_y+triple_subject_h, triple_subject_x:triple_subject_x+triple_subject_w]
                img_sbj=cv2.resize(img_sbj,(224,224))
                img_sbj=img_sbj.transpose(2, 0, 1)-MEAN_VALUES
                img_obj=img[triple_object_y:triple_object_y+triple_object_h, triple_object_x:triple_object_x+triple_object_w]
                img_obj=cv2.resize(img_obj,(224,224))
                img_obj=img_obj.transpose(2, 0, 1)-MEAN_VALUES
            except Exception as e:
                print 'image reading error'
                print 'type:' + str(type(e))
                print 'args:' + str(e.args)
                print 'message:' + e.message.strip()
                print 'image_id:'+ str(image_id)
                print 'triple_id:'+ str(triple_id)
                print ""
                continue
            triple_id2relation_id[triple_id]=relation_id
            all_img_list.append(img_sbj)
            all_img_list.append(img_obj)
            triple_ids_tmp.append(triple_id)
        
        if len(all_img_list)>0:
            x_batch = np.array(all_img_list, dtype=np.float32)
            if gpu_id >=0:
                x = Variable(cuda.to_gpu(x_batch), volatile=True)
            else:
                x = Variable(x_batch, volatile=True)
            image_feature_chainer, = func(inputs={'data': x}, outputs=['pool5/7x7_s1'],
                          disable=['loss1/ave_pool', 'loss2/ave_pool','loss3/classifier'],
                          train=False)

            shape=image_feature_chainer.data.shape[0:2]
            image_feature_xp=image_feature_chainer.data.reshape(shape)
            image_feature_np=cuda.to_cpu(image_feature_xp)

            for i in xrange(image_feature_np.shape[0]/2):
                triple_id=triple_ids_tmp[i]
                triple_id2subject_feature[triple_id]=image_feature_np[2*i]
                triple_id2object_feature[triple_id]=image_feature_np[2*i+1]

    return triple_id2subject_feature, triple_id2object_feature,triple_id2relation_id

with open('../data/relationships-half.json', 'r') as f:
    all_relationships = json.load(f)

with open('../work/index2relation.pkl', 'r') as f:
    index2relation=pickle.load(f)
relation2index = dict((v, k) for k, v in index2relation.iteritems())

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

triple_id2subject_feature, triple_id2object_feature,triple_id2relation_id=pre_extract_relation_feature(relationship_train_image_ids)
with open('../work/relationship_train_triple_features.cpickle', 'w') as f:
    cpickle.dump((triple_id2subject_feature, triple_id2object_feature,triple_id2relation_id),f)

triple_id2subject_feature, triple_id2object_feature,triple_id2relation_id=pre_extract_relation_feature(relationship_val_image_ids)
with open('../work/relationship_val_triple_features.cpickle', 'w') as f:
    cpickle.dump((triple_id2subject_feature, triple_id2object_feature,triple_id2relation_id),f)

triple_id2subject_feature, triple_id2object_feature,triple_id2relation_id=pre_extract_relation_feature(relationship_test_image_ids)
with open('../work/relationship_test_triple_features.cpickle', 'w') as f:
    cpickle.dump((triple_id2subject_feature, triple_id2object_feature,triple_id2relation_id),f)
