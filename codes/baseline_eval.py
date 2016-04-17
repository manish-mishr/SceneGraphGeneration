#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Author: satoshi tsustsui

This is a code for evaluating baseline for secne graph via caption.

'''

import cPickle as pickle
import json

with open('../work/relationships_baseline_test.cpickle', 'r') as f:
    relationships_baseline_test=pickle.load(f)

with open('../work/attributes_baseline_test.cpickle', 'r') as f:
    attributes_baseline_test=pickle.load(f)

with open('../data/relationships-half.json', 'r') as f:
    all_relationships = json.load(f)

with open('../data/attributes-half.json', 'r') as f:
    all_attributes = json.load(f)

def get_image_relation_triples(image_id,all_relationships):
    for i in xrange(len(all_relationships)):
        image_id_=all_relationships[i][u'id']
        if image_id==image_id_:
            relationships=all_relationships[i][u'relationships']
    image_relationship_triples={}
    for triple in relationships:
        triple_id=triple[u'id']
        triple_predicate=triple[u'predicate']
        triple_subject_name=triple[u'subject']['name']
        triple_object_name=triple[u'object']['name']
        image_relationship_triples[triple_id]=[triple_subject_name,triple_predicate,triple_object_name]
    return image_relationship_triples

def get_image_attribute_triples(image_id,all_attributes):
    for i in xrange(len(all_attributes)):
        image_id_=all_attributes[i][u'id']
        if image_id==image_id_:
            attributes=all_attributes[i][u'attributes']
    image_attribute_triples=[]
    for triple in attributes:
        triple_id=triple[u'id']
        triple_predicate="is"
        triple_subject=str(triple[u'object_names'][0])#we ignore mulitple object namae...
        for attribute in triple[u'attributes']:
            image_attribute_triples.append([triple_subject,triple_predicate,attribute])        
    return image_attribute_triples

def jaccard(s1, s2):
	s1=[ "-".join(triple) for triple in s1]
	s2=[ "-".join(triple) for triple in s2]
	st1=set(s1)
	st2=set(s2)
	u = set(st1).union(st2)
	i = set(st1).intersection(st2)
	return float(len(i))/float(len(u))

relationships_baseline_test_results={}
for (image_id,triples) in relationships_baseline_test.iteritems():
	ground_truth_triples=get_image_relation_triples(image_id,all_relationships)
	jaccad_index=jaccard(triples,ground_truth_triples.values())
	relationships_baseline_test_results[image_id]=jaccad_index

#[x for x in relationships_baseline_test_results.values() if x >0]
#zero

attributes_baseline_test_results={}
for (image_id,triples) in attributes_baseline_test.iteritems():
	ground_truth_triples=get_image_attribute_triples(image_id,all_attributes)
	jaccad_index=jaccard(triples,ground_truth_triples)
	attributes_baseline_test_results[image_id]=jaccad_index

#[x for x in attributes_baseline_test_results.values() if x >0]
#zero

with open('../work/attributes_baseline_test_jaccard.json', 'w') as f:
    json.dump(attributes_baseline_test_results,f)

with open('../work/relationships_baseline_test_jaccard.json', 'w') as f:
    json.dump(relationships_baseline_test_results,f)