from copy import deepcopy
import id3new
def prune(testInstances,tree):

	#we get a tree as input parameter

	bestPrecision = deepcopy(tree.precision)
	bestNodes = deepcopy(tree.nodes)
	bestTree = tree
	#now do the following:
	for node in bestTree.nodes:
		if node not in bestTree.leafNodes:
			#set as a leaf
			node.leafOrNot = 1
			#now check the new accuracy with this current tree
			bestTree = id3new.testTree(testInstances,bestTree)
			if newTree.precision > bestPrecision:
				#then reset that node to 0
				node.leafOrNot = 0
			else:
				bestPrecision = bestTree.precision
				print 'found a better solution with accuracy: ',bestPrecision

	return bestTree