import sys
import math

def prune(tree):
	for node in nodes:
		#check the accuracy of the validation set on the tree with and without each node that is not a leaf
		#if accuracy of the validation set increases or stays the same by dropping the node, then drop the node