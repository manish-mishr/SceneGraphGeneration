#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Author: satoshi tsustsui

This is a sample code that generates scene graph from image via caption.
Pipline will be: 

image -> caption -> scene_graph.

image -> caption is done by https://github.com/apple2373/chainer_caption_generation/
This git repo is also made by Satoshi Tsutsui.

caption -> scene_graph is done by http://nlp.stanford.edu/software/scenegraph-parser.shtml
This is the progeam developed by Stanford NLP lab. I call their java program by python.
This use py4j to call java from python. Java 1.8+ and Python 2.7x is required.
You should set up parser in advance by the following command: 
java -jar SecneGraphParserPython.jar 

'''
import sys
sys.path.append('../')

import numpy as np
from chainer_caption_generation.codes.image_reader import Image_reader
from chainer_caption_generation.codes.caption_generator import Caption_generator

#make sure the java commad is excuted in adcace
from py4j.java_gateway import JavaGateway
gateway = JavaGateway()
parser = gateway.entry_point.getGraphGenerator() 

#Instantiate image_reader with GoogleNet mean image
mean_image = np.array([104, 117, 123]).reshape((3,1,1))
image_reader=Image_reader(mean=mean_image)

#Instantiate caption generator
caption_model_place='../chainer_caption_generation/models/caption_model.chainer'
cnn_model_place='../chainer_caption_generation/data/bvlc_googlenet_caffe_chainer.pkl'
index2word_place='../chainer_caption_generation/work/index2token.pkl'
caption_generator=Caption_generator(caption_model_place=caption_model_place,cnn_model_place=cnn_model_place,index2word_place=index2word_place)

#read an image as numpy arrays
image_file_path='../chainer_caption_generation/images/COCO_val2014_000000185546.jpg'
image=image_reader.read(image_file_path)
caption=caption_generator.get_top_sentence(image)
parser.parse_caption(caption)

print "caption"
print caption
print "scene graph"
print parser.get_relations()
print parser.get_attributes()