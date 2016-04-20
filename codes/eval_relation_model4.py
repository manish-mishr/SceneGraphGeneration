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
model_dir='../experiment4/'
savedir=model_dir

#Override Settings by argument
parser = argparse.ArgumentParser(description=u"caption generation")
parser.add_argument("-g", "--gpu",default=gpu_id, type=int, help=u"GPU ID.CPU is -1")
parser.add_argument("-m", "--modeldir",default=model_dir, type=str, help=u"The directory that have models")
args = parser.parse_args()
gpu_id=args.gpu
model_dir= args.modeldir

#Gpu Setting
if gpu_id >= 0:
    xp = cuda.cupy 
    cuda.get_device(gpu_id).use()
else:
    xp=np

#Prepare Data
print("loading preprocessed data")

with open('../work/relationship_val_features.cpickle', 'r') as f:
    image_id2image_feature=cpickle.load(f)

with open('../work/relationship_val_triple_features.cpickle', 'r') as f:
    triple_id2subject_feature, triple_id2object_feature,triple_id2relation_id=cpickle.load(f)

with open('../work/relationship_test_features.cpickle', 'r') as f:
    image_id2image_feature_test=cpickle.load(f)

with open('../work/relationship_test_triple_features.cpickle', 'r') as f:
    triple_id2subject_feature_test, triple_id2object_feature_test,triple_id2relation_id_test=cpickle.load(f)

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
model.h3=F.Linear(n_units, n_units)#hidden unit,#parameter  W,b
model.bn3=F.BatchNormalization(n_units)#parameter  gamma,beta
model.h3=F.Linear(n_units, n_units)#hidden unit,#parameter  W,b
model.bn3=F.BatchNormalization(n_units)#parameter  gamma,beta
model.h4=F.Linear(n_units, n_units)#hidden unit,#parameter  W,b
model.bn4=F.BatchNormalization(n_units)#parameter  gamma,beta
model.h5=F.Linear(n_units, n_units)#hidden unit,#parameter  W,b
model.bn5=F.BatchNormalization(n_units)#parameter  gamma,beta
model.h6=F.Linear(n_units, n_units)#hidden unit,#parameter  W,b
model.bn6=F.BatchNormalization(n_units)#parameter  gamma,beta
model.h7=F.Linear(n_units, n_units)#hidden unit,#parameter  W,b
model.bn7=F.BatchNormalization(n_units)#parameter  gamma,beta
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
    l2 = F.relu(model.bn2(model.h2(l1)))
    l3 = F.relu(model.bn3(model.h3(l2)))
    l4 = F.relu(model.bn4(model.h4(l3)))
    l5 = F.relu(model.bn5(model.h5(l4)))
    l6 = F.relu(model.bn6(model.h6(l5)))
    l7 = F.relu(model.bn7(model.h7(l6)))
    y = model.out(l7)
    loss=F.softmax_cross_entropy(y, t)
    accuracy=F.accuracy(y, t)
    return loss,accuracy

#Validiation Setting
batchsize=1024
num_train_data=len(triple_id2subject_feature)
all_triple_ids=triple_id2subject_feature.keys()

#Begin Validiating
loss_list=[]
accuracy_list=[]
for i in xrange(200):
    model_place=model_dir+'/relation_model%d.chainer'%i
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

        loss, accuracy = forward(x_batch, y_batch, train=False)
        sum_loss     += loss.data * batchsize
        sum_accuracy += accuracy.data * batchsize

    mean_loss     = sum_loss / num_train_data
    mean_accuracy = sum_accuracy / num_train_data

    print mean_loss,mean_accuracy
    accuracy_list.append(mean_accuracy)
    loss_list.append(mean_loss)
    with open(savedir+"val_mean_loss.txt", "a") as f:
        f.write(str(mean_loss)+'\n')
    with open(savedir+"val_mean_accuracy.txt", "a") as f:
        f.write(str(mean_accuracy)+'\n')


#Test evaluation
triple_id2subject_feature, triple_id2object_feature,triple_id2relation_id=triple_id2subject_feature_test, triple_id2object_feature_test,triple_id2relation_id_test
image_id2image_feature=image_id2image_feature_test
batchsize=1024
num_train_data=len(triple_id2subject_feature)
all_triple_ids=triple_id2subject_feature.keys()

argmax=np.argmax(accuracy_list)
model_place=model_dir+'/relation_model%d.chainer'%argmax
print model_place
serializers.load_hdf5(model_place, model)#load modeldir


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

    loss, accuracy = forward(x_batch, y_batch, train=False)
    sum_loss     += loss.data * batchsize
    sum_accuracy += accuracy.data * batchsize

mean_loss     = sum_loss / num_train_data
mean_accuracy = sum_accuracy / num_train_data

print "test loss and test accuracy"
print mean_loss,mean_accuracy
with open(savedir+"tes_mean_loss_accuracy.txt", "w") as f:
    f.write(str(mean_loss)+","+str(mean_accuracy)+'\n')
