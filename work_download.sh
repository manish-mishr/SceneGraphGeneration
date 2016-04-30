#! /bin/bash
cd work
if [ ! -f attribute_dic_freq.txt ]; then
	wget https://googledrive.com/host/0B046sNk0DhCDbmZrX1VYWDBXOUE/attribute_dic_freq.txt
fi
if [ ! -f attributes_baseline_test.cpickle ]; then
	wget https://googledrive.com/host/0B046sNk0DhCDbmZrX1VYWDBXOUE/attributes_baseline_test.cpickle
fi
if [ ! -f attributes_baseline_test_jaccard.json ]; then
	wget https://googledrive.com/host/0B046sNk0DhCDbmZrX1VYWDBXOUE/attributes_baseline_test_jaccard.json
fi
if [ ! -f attributes_baseline_test.json ]; then
	wget https://googledrive.com/host/0B046sNk0DhCDbmZrX1VYWDBXOUE/attributes_baseline_test.json
fi
if [ ! -f attribute_test_image_ids.csv ]; then
	wget https://googledrive.com/host/0B046sNk0DhCDbmZrX1VYWDBXOUE/attribute_test_image_ids.csv
fi
if [ ! -f attribute_test_triple_features.cpickle ]; then
	wget https://googledrive.com/host/0B046sNk0DhCDbmZrX1VYWDBXOUE/attribute_test_triple_features.cpickle
fi
if [ ! -f attribute_train_image_ids.csv ]; then
	wget https://googledrive.com/host/0B046sNk0DhCDbmZrX1VYWDBXOUE/attribute_train_image_ids.csv
fi
if [ ! -f attribute_train_triple_features.cpickle ]; then
	wget https://googledrive.com/host/0B046sNk0DhCDbmZrX1VYWDBXOUE/attribute_train_triple_features.cpickle
fi
if [ ! -f attribute_val_image_ids.csv ]; then
	wget https://googledrive.com/host/0B046sNk0DhCDbmZrX1VYWDBXOUE/attribute_val_image_ids.csv
fi
if [ ! -f attribute_val_triple_features.cpickle ]; then
	wget https://googledrive.com/host/0B046sNk0DhCDbmZrX1VYWDBXOUE/attribute_val_triple_features.cpickle
fi
if [ ! -f attr_triple_id2image_id.pkl ]; then
	wget https://googledrive.com/host/0B046sNk0DhCDbmZrX1VYWDBXOUE/attr_triple_id2image_id.pkl
fi
if [ ! -f index2attribute.pkl ]; then
	wget https://googledrive.com/host/0B046sNk0DhCDbmZrX1VYWDBXOUE/index2attribute.pkl
fi
if [ ! -f index2relation.pkl ]; then
	wget https://googledrive.com/host/0B046sNk0DhCDbmZrX1VYWDBXOUE/index2relation.pkl
fi
if [ ! -f relation_dic_freq.txt ]; then
	wget https://googledrive.com/host/0B046sNk0DhCDbmZrX1VYWDBXOUE/relation_dic_freq.txt
fi
if [ ! -f relationships_baseline_test.cpickle ]; then
	wget https://googledrive.com/host/0B046sNk0DhCDbmZrX1VYWDBXOUE/relationships_baseline_test.cpickle
fi
if [ ! -f relationships_baseline_test_jaccard.json ]; then
	wget https://googledrive.com/host/0B046sNk0DhCDbmZrX1VYWDBXOUE/relationships_baseline_test_jaccard.json
fi
if [ ! -f relationships_baseline_test.json ]; then
	wget https://googledrive.com/host/0B046sNk0DhCDbmZrX1VYWDBXOUE/relationships_baseline_test.json
fi
if [ ! -f relationship_test_features.cpickle ]; then
	wget https://googledrive.com/host/0B046sNk0DhCDbmZrX1VYWDBXOUE/relationship_test_features.cpickle
fi
if [ ! -f relationship_test_image_ids.csv ]; then
	wget https://googledrive.com/host/0B046sNk0DhCDbmZrX1VYWDBXOUE/relationship_test_image_ids.csv
fi
if [ ! -f relationship_test_triple_features.cpickle ]; then
	wget https://googledrive.com/host/0B046sNk0DhCDbmZrX1VYWDBXOUE/relationship_test_triple_features.cpickle
fi
if [ ! -f relationship_train_features.cpickle ]; then
	wget https://googledrive.com/host/0B046sNk0DhCDbmZrX1VYWDBXOUE/relationship_train_features.cpickle
fi
if [ ! -f relationship_train_image_ids.csv ]; then
	wget https://googledrive.com/host/0B046sNk0DhCDbmZrX1VYWDBXOUE/relationship_train_image_ids.csv
fi
if [ ! -f relationship_train_triple_features.cpickle ]; then
	wget https://googledrive.com/host/0B046sNk0DhCDbmZrX1VYWDBXOUE/relationship_train_triple_features.cpickle
fi
if [ ! -f relationship_val_features.cpickle ]; then
	wget https://googledrive.com/host/0B046sNk0DhCDbmZrX1VYWDBXOUE/relationship_val_features.cpickle
fi
if [ ! -f relationship_val_image_ids.csv ]; then
	wget https://googledrive.com/host/0B046sNk0DhCDbmZrX1VYWDBXOUE/relationship_val_image_ids.csv
fi
if [ ! -f relationship_val_triple_features.cpickle ]; then
	wget https://googledrive.com/host/0B046sNk0DhCDbmZrX1VYWDBXOUE/relationship_val_triple_features.cpickle
fi
if [ ! -f rel_triple_id2image_id.pkl ]; then
	wget https://googledrive.com/host/0B046sNk0DhCDbmZrX1VYWDBXOUE/rel_triple_id2image_id.pkl
fi
