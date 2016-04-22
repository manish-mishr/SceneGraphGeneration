#!/usr/bin/env python
# -*- coding: utf-8 -*-


import os
os.environ["CHAINER_TYPE_CHECK"] = "0" #to disable type check. 
import chainer 
#Check che below is False if you disabled type check
#print(chainer.functions.Linear(1,1).type_check_enable) 

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
savedir='../experiment1a/'# name of log and results image saving directory

#Override Settings by argument
parser = argparse.ArgumentParser(description=u"caption generation")
parser.add_argument("-g", "--gpu",default=gpu_id, type=int, help=u"GPU ID.CPU is -1")
parser.add_argument("-d", "--savedir",default=savedir, type=str, help=u"The directory to save models and log")
args = parser.parse_args()
gpu_id=args.gpu
savedir=args.savedir
if not os.path.exists(savedir):
    os.makedirs(savedir)
model_dir=savedir


#Gpu Setting
if gpu_id >= 0:
    xp = cuda.cupy 
    cuda.get_device(gpu_id).use()
else:
    xp=np

#Prepare Data
print("loading preprocessed data")

with open('../work/attribute_val_features.cpickle', 'r') as f:
    image_id2image_feature=cpickle.load(f)

with open('../work/attribute_val_triple_features.cpickle', 'r') as f:
    triple_id2subject_feature,triple_id2attribute_ids=cpickle.load(f)

with open('../work/attribute_test_features.cpickle', 'r') as f:
    image_id2image_test_feature=cpickle.load(f)

with open('../work/attribute_test_triple_features.cpickle', 'r') as f:
    triple_id2subject_test_feature,triple_id2attribute_test_ids=cpickle.load(f)

with open('../work/attr_triple_id2image_id.pkl', 'r') as f:
    attr_triple_id2image_id=pickle.load(f)

with open('../work/index2attribute.pkl', 'r') as f:
    index2attribute=pickle.load(f)
attribute2index = dict((v, k) for k, v in index2attribute.iteritems())

#Model Preparation
print "preparing model"
image_feature_dim=1024#image feature dimention per image
n_units = 128  # number of units per layer
vocab_size=len(attribute2index)

model = chainer.FunctionSet()
model.img_feature2vec=F.Linear(2*image_feature_dim, n_units)#parameter  W,b
model.bn_feature=F.BatchNormalization(n_units)#parameter  sigma,gamma
model.h1=F.Linear(n_units, n_units)#hidden unit,#parameter  W,b
model.bn1=F.BatchNormalization(n_units)#parameter  gamma,beta
model.out=F.Linear(n_units, vocab_size)#parameter  W,b

#To GPU
if gpu_id >= 0:
    model.to_gpu()

#Define Newtowork (Forward)
def forward(x_data, y_data,train=True):
    x = Variable(x_data,volatile=not train)
    t = Variable(y_data,volatile=not train)
    feature_input = F.relu(model.bn_feature(model.img_feature2vec(x)))
    l1 = F.relu(model.bn1(model.h1(feature_input)))
    y = model.out(l1)
    loss=F.sigmoid_cross_entropy(y, t)

    predicted=cuda.to_cpu(F.sigmoid(y).data)
    predicted[predicted>0.5]=1
    predicted[predicted<=0.5]=0
    label=cuda.to_cpu(y_data)
    index=np.multiply(predicted==label,label==1)
    accuracy=float(np.sum(index))/float(np.sum(label==1))

    return loss,accuracy

#Trining Setting
batchsize=512
num_train_data=len(triple_id2subject_feature)
all_triple_ids=triple_id2subject_feature.keys()

#Begin Validiating
loss_list=[]
accuracy_list=[]
print 'training started'
for epoch in xrange(200):
    print 'epoch %d' %epoch

    model_place=model_dir+'/attribute_model%d.chainer'%epoch
    print model_place
    if os.path.exists(model_place):
        serializers.load_hdf5(model_place, model)#load modeldir
    else:
        continue

    sum_loss = 0
    sum_accuracy=0
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
            image_id=attr_triple_id2image_id[triple_id]
            image_feature=image_id2image_feature[image_id]

            #make concatnated vector
            vec=np.zeros([2*image_feature_dim],dtype=np.float32)
            vec[0:image_feature_dim]=image_feature
            vec[image_feature_dim:2*image_feature_dim]=subject_feature

            attr_ids=triple_id2attribute_ids[triple_id]
            y_vec=np.zeros(vocab_size)
            y_vec[attr_ids]=1

            x_batch_list.append(vec)
            y_batch_list.append(y_vec)
        
        x_batch=np.array(x_batch_list,dtype=np.float32)
        y_batch=np.array(y_batch_list,dtype=np.int32)     
        if gpu_id >= 0:
            x_batch = cuda.to_gpu(x_batch)
            y_batch = cuda.to_gpu(y_batch)

        loss, accuracy = forward(x_batch, y_batch,train=False)
        #print loss.data,accuracy.data
        sum_loss     += loss.data * batchsize
        sum_accuracy += accuracy * batchsize

    mean_loss     = sum_loss / num_train_data
    mean_accuracy = sum_accuracy / num_train_data
    print mean_loss,mean_accuracy
    loss_list.append(mean_loss)
    accuracy_list.append(mean_accuracy)
    with open(savedir+"val_mean_loss.txt", "a") as f:
        f.write(str(mean_loss)+'\n')
    with open(savedir+"val_mean_accuracy.txt", "a") as f:
        f.write(str(mean_accuracy)+'\n')

#testing
image_id2image_feature=image_id2image_test_feature
triple_id2subject_feature,triple_id2attribute_ids=triple_id2subject_test_feature,triple_id2attribute_test_ids
#Trining Setting
batchsize=512
num_train_data=len(triple_id2subject_feature)
all_triple_ids=triple_id2subject_feature.keys()

argmax=np.argmax(accuracy_list)
model_place=model_dir+'/attribute_model%d.chainer'%argmax
print model_place
serializers.load_hdf5(model_place, model)#load model_dir

sum_loss = 0
sum_accuracy=0
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
        image_id=attr_triple_id2image_id[triple_id]
        image_feature=image_id2image_feature[image_id]

        #make concatnated vector
        vec=np.zeros([2*image_feature_dim],dtype=np.float32)
        vec[0:image_feature_dim]=image_feature
        vec[image_feature_dim:2*image_feature_dim]=subject_feature

        attr_ids=triple_id2attribute_ids[triple_id]
        y_vec=np.zeros(vocab_size)
        y_vec[attr_ids]=1

        x_batch_list.append(vec)
        y_batch_list.append(y_vec)
    
    x_batch=np.array(x_batch_list,dtype=np.float32)
    y_batch=np.array(y_batch_list,dtype=np.int32)     
    if gpu_id >= 0:
        x_batch = cuda.to_gpu(x_batch)
        y_batch = cuda.to_gpu(y_batch)

    loss, accuracy = forward(x_batch, y_batch,train=False)
    #print loss.data,accuracy.data

    sum_loss     += loss.data * batchsize
    sum_accuracy += accuracy * batchsize

mean_loss     = sum_loss / num_train_data
mean_accuracy = sum_accuracy / num_train_data

print "test loss and test accuracy"
print mean_loss,mean_accuracy
with open(savedir+"test_mean_loss_accuracy.txt", "w") as f:
    f.write(str(mean_loss)+","+str(mean_accuracy)+'\n')