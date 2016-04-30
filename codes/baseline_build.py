#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Author: satoshi tsustsui

This is a code for building baseline for secne graph via caption.

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
import cPickle as pickle
import json
import copy

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
caption_generator=Caption_generator(caption_model_place=caption_model_place,cnn_model_place=cnn_model_place,index2word_place=index2word_place,gpu_id=0)

import unicodecsv as csv

with open('../work/relationship_test_image_ids.csv', 'r') as f:
    reader = csv.reader(f)
    relationship_test_image_ids=reader.next()
    relationship_test_image_ids = map(int,relationship_test_image_ids)

with open('../work/attribute_test_image_ids.csv', 'r') as f:
    reader = csv.reader(f)
    attribute_test_image_ids=reader.next()
    attribute_test_image_ids = map(int,attribute_test_image_ids)

def copy_triples(triples_java):
	python_triples=[]
	for triple in triples_java:
		python_triple=[]
		for element in triple:
			python_triple.append(copy.deepcopy(element))
		python_triples.append(python_triple)
	return python_triples

relationships_baseline_test={}
for image_id in relationship_test_image_ids:
	relations_in_image={}
	image_file_path="../data/images/%d.jpg"%image_id
	image=image_reader.read(image_file_path)
	caption=caption_generator.get_top_sentence(image)
	relations_in_image["caption"]=caption
	parser.parse_caption(caption)
	relations=parser.get_relations()
	relations_in_image["relations"]=copy_triples(relations)
	relationships_baseline_test[image_id]=relations_in_image

attributes_baseline_test={}
for image_id in attribute_test_image_ids:
	attributes_in_image={}
	image_file_path="../data/images/%d.jpg"%image_id
	image=image_reader.read(image_file_path)
	caption=caption_generator.get_top_sentence(image)
	attributes_in_image["caption"]=caption
	parser.parse_caption(caption)
	attributes=parser.get_attributes()
	attributes_in_image["attributes"]=copy_triples(attributes)
	attributes_baseline_test[image_id]=attributes_in_image

with open('../work/relationships_baseline_test.cpickle', 'w') as f:
    pickle.dump(relationships_baseline_test,f)

with open('../work/attributes_baseline_test.cpickle', 'w') as f:
    pickle.dump(attributes_baseline_test,f)

with open('../work/attributes_baseline_test.json', 'w') as f:
    json.dump(attributes_baseline_test,f)

with open('../work/relationships_baseline_test.json', 'w') as f:
    json.dump(relationships_baseline_test,f)
