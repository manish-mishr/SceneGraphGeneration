#!/usr/bin/env python
# -*- coding: utf-8 -*-


import os
os.environ["CHAINER_TYPE_CHECK"] = "0" #to disable type check. 
import chainer 
#Check che below is False if you disabled type check
#print(chainer.functions.Linear(1,1).type_check_enable) 

import os


import chainer.functions as F
from chainer import cuda
from chainer import Function, FunctionSet, Variable, optimizers, serializers

import argparse
import numpy as np
import cPickle as cpickle
import pickle
import random
import unicodecsv as csv
from tqdm import tqdm
import cv2

#Settings can be changed by command line arguments
gpu_id=0# GPU ID. if you want to use cpu, -1
#gpu_id=4
savedir='../experiment2/'# name of log and results image saving directory

#Override Settings by argument
parser = argparse.ArgumentParser(description=u"caption generation")
parser.add_argument("-g", "--gpu",default=gpu_id, type=int, help=u"GPU ID.CPU is -1")
parser.add_argument("-d", "--savedir",default=savedir, type=str, help=u"The directory to save models and log")
args = parser.parse_args()
gpu_id=args.gpu
savedir=args.savedir
if not os.path.exists(savedir):
    os.makedirs(savedir)

#Gpu Setting
if gpu_id >= 0:
    xp = cuda.cupy 
    cuda.get_device(gpu_id).use()
else:
    xp=np

#Prepare Data
print("loading preprocessed data")

with open('../work/relationship_train_image_ids.csv', 'r') as f:
    reader = csv.reader(f)
    relationship_train_image_ids=reader.next()
    relationship_train_image_ids = map(int,relationship_train_image_ids)

with open('../work/relationship_train_features.cpickle', 'r') as f:
    image_id2image_feature=cpickle.load(f)

with open('../work/relationship_train_triple_features.cpickle', 'r') as f:
    triple_id2subject_feature, triple_id2object_feature,triple_id2relation_id=cpickle.load(f)

with open('../work/rel_triple_id2image_id.pkl', 'r') as f:
    rel_triple_id2image_id=pickle.load(f)

with open('../work/index2relation.pkl', 'r') as f:
    index2relation=pickle.load(f)
relation2index = dict((v, k) for k, v in index2relation.iteritems())

#Model Preparation
print "preparing model"
image_feature_dim=1024#image feature dimention per image
n_units = 1024  # number of units per layer
vocab_size=len(relation2index)

model = chainer.FunctionSet()
model.img_feature2vec=F.Linear(3*image_feature_dim, n_units)#parameter  W,b
model.bn_feature=F.BatchNormalization(n_units)#parameter  sigma,gamma
model.h1=F.Linear(n_units, n_units)#hidden unit,#parameter  W,b
model.bn1=F.BatchNormalization(n_units)#parameter  gamma,beta
model.h2=F.Linear(n_units, n_units)#hidden unit,#parameter  W,b
model.bn2=F.BatchNormalization(n_units)#parameter  gamma,beta
model.out=F.Linear(n_units, vocab_size)#parameter  W,b

#Parameter Initialization
for param in model.params():
    data = param.data
    data[:] = np.random.uniform(-0.1, 0.1, data.shape)

#To GPU
if gpu_id >= 0:
    model.to_gpu()

#Define Newtowork (Forward)
def forward(x_data, y_data,train=True):
    x = Variable(x_data,volatile=not train)
    t = Variable(y_data,volatile=not train)
    feature_input = F.relu(model.bn_feature(model.img_feature2vec(x)))
    l1 = F.relu(model.bn1(model.h1(feature_input)))
    l2 = F.relu(model.bn2(model.h2(l1)))
    y = model.out(l2)
    loss=F.softmax_cross_entropy(y, t)
    accuracy=F.accuracy(y, t)
    return loss,accuracy

optimizer = optimizers.Adam()
optimizer.setup(model)

#Trining Setting
batchsize=512
num_train_data=len(triple_id2subject_feature)
all_triple_ids=triple_id2subject_feature.keys()

#Begin Training
print 'training started'
for epoch in xrange(200):
    print 'epoch %d' %epoch
    sum_loss = 0
    sum_accuracy=0
    random.shuffle(all_triple_ids)
    for i in tqdm(xrange(0, num_train_data, batchsize)):
        x_batch_list=[]
        y_batch_list=[]
        for j in xrange(batchsize):
            #get ids and features
            if i+j < num_train_data:
                triple_id=all_triple_ids[i+j]
            else:
                break

            subject_feature=triple_id2subject_feature[triple_id]
            object_feature=triple_id2object_feature[triple_id]
            image_id=rel_triple_id2image_id[triple_id]
            image_feature=image_id2image_feature[image_id]

            #make concatnated vector
            vec=np.zeros([3*image_feature_dim],dtype=np.float32)
            vec[0:image_feature_dim]=image_feature
            vec[image_feature_dim:2*image_feature_dim]=subject_feature
            vec[2*image_feature_dim:3*image_feature_dim]=object_feature

            rel_id=triple_id2relation_id[triple_id]

            x_batch_list.append(vec)
            y_batch_list.append(rel_id)
        
        x_batch=np.array(x_batch_list,dtype=np.float32)
        y_batch=np.array(y_batch_list,dtype=np.int32)     
        if gpu_id >= 0:
            x_batch = cuda.to_gpu(x_batch)
            y_batch = cuda.to_gpu(y_batch)

        optimizer.zero_grads()
        loss, accuracy = forward(x_batch, y_batch)
        #print loss.data,accuracy.data
        with open(savedir+"real_loss.txt", "a") as f:
            f.write(str(loss.data)+'\n')
        with open(savedir+"real_accuracy.txt", "a") as f:
            f.write(str(accuracy.data)+'\n') 
        loss.backward()
        optimizer.update()     
        sum_loss     += loss.data * batchsize
        sum_accuracy += accuracy.data * batchsize
    
    serializers.save_hdf5(savedir+"/caption_model"+str(epoch)+'.chainer', model)
    serializers.save_hdf5(savedir+"/optimizer"+str(epoch)+'.chainer', optimizer)

    mean_loss     = sum_loss / num_train_data
    mean_accuracy = sum_accuracy / num_train_data
    print mean_loss,mean_accuracy
    with open(savedir+"mean_loss.txt", "a") as f:
        f.write(str(mean_loss)+'\n')
    with open(savedir+"mean_accuracy.txt", "a") as f:
        f.write(str(mean_accuracy)+'\n')
