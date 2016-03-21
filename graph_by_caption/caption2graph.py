
'''
Author: satoshi tsustsui

Just a sample to parse caption to generate scene graph

The parser reference: http://nlp.stanford.edu/software/scenegraph-parser.shtml
This use py4j to call java from python.
Java 1.8+ is required.
Python 2.7x is required.

You should set up parser in advance by the following command: 
java -jar SecneGraphParserPython.jar 

Then you can excute this file. 
'''

from py4j.java_gateway import JavaGateway
gateway = JavaGateway()
parser = gateway.entry_point.getGraphGenerator() 

cap="a white plate topped with a slice of cake"
parser.parse_caption(cap)
print cap
print parser.get_relations()
print parser.get_attributes()

cap="a cat is sitting on a black chair in a room"
parser.parse_caption(cap)
print cap
print parser.get_relations()
print parser.get_attributes()