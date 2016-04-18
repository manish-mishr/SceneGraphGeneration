#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
To extarct CNN features.

'''


import os
os.environ["CHAINER_TYPE_CHECK"] = "0" #to disable type check
import chainer 
import argparse
import numpy as np
from chainer import cuda
import chainer.functions as F
from chainer.functions import caffe
from chainer import cuda, Function, FunctionSet, gradient_check, Variable, optimizers
import json
import pickle
import cPickle as cpickle
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

def get_image_attributes(image_id,all_attributes):
    for i in xrange(len(all_attributes)):
        image_id_=all_objects[i][u'id']
        if image_id==image_id_:
            attributes=all_attributes[i][u'attributes']
    return attributes

with open('../data/attributes-half.json', 'r') as f:
    all_attributes = json.load(f)

with open('../data/objects.json', 'r') as f:
    all_objects = json.load(f)

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

with open('../work/index2attribute.pkl', 'r') as f:
    index2attribute=pickle.load(f)
attribute2index = dict((v, k) for k, v in index2attribute.iteritems())

def pre_extract_attribute_feature(image_ids):
    triple_id2subject_feature={}
    triple_id2attribute_ids={}
    for image_id in tqdm(image_ids):
        image_objects=get_image_attributes(image_id,all_attributes)
        img_file="../data/images/%d.jpg"%image_id
        img=cv2.imread(img_file,cv2.IMREAD_COLOR)
        all_img_list=[]
        triple_ids_tmp=[]
        for image_object in image_objects:
            object_x=image_object[u'x']
            object_y=image_object[u'y']
            object_w=image_object[u'w']
            object_h=image_object[u'h']
            triple_id=image_object[u'id']
            triple_attritbues=image_object[u'attributes']
            try:
                img_sbj=img[object_y:object_y+object_h, object_x:object_x+object_w]
                img_sbj=cv2.resize(img_sbj,(224,224))
                img_sbj=img_sbj.transpose(2, 0, 1)-MEAN_VALUES
            except Exception as e:
                print e
                continue
            
            attribute_ids=[]
            for attribute in triple_attritbues:
                if attribute not in attribute2index:
                    continue
                attribute_id=attribute2index[attribute]
                attribute_ids.append(attribute_id)
                
            if len(attribute_ids) >0:
                triple_id2attribute_ids[triple_id]=attribute_ids
                all_img_list.append(img_sbj)
                triple_ids_tmp.append(triple_id)
            else:
                continue
        try:
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
                
                for i in xrange(image_feature_np.shape[0]):
                    triple_id=triple_ids_tmp[i]
                    triple_id2subject_feature[triple_id]=image_feature_np[i]
                
        except Exception as e:
            print e
            continue
                
    return triple_id2subject_feature,triple_id2attribute_ids

triple_id2subject_feature,triple_id2attribute_ids=pre_extract_attribute_feature(attribute_train_image_ids)
with open('../work/attribute_train_triple_features.cpickle', 'w') as f:
    cpickle.dump((triple_id2subject_feature,triple_id2attribute_ids),f)

triple_id2subject_feature,triple_id2attribute_ids=pre_extract_attribute_feature(attribute_val_image_ids)
with open('../work/attribute_val_triple_features.cpickle', 'w') as f:
    cpickle.dump((triple_id2subject_feature,triple_id2attribute_ids),f)

triple_id2subject_feature,triple_id2attribute_ids=pre_extract_attribute_feature(attribute_test_image_ids)
with open('../work/attribute_test_triple_features.cpickle', 'w') as f:
    cpickle.dump((triple_id2subject_feature,triple_id2attribute_ids),f)
