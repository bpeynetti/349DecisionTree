import sys
import math
from copy import deepcopy
import  matplotlib.pyplot as plt
import random
import csv


class ID3Tree(object):
	def __init__(self):
		self.data = None
		self.attr_dict = None
		self.attributeList = None
		self.attrType = None
		self.root = None
		self.attributeNumbers = 0
		self.attrNum = None

class TreeNode(object):
	def __init__(self):
		self.data=None
		self.attributes = None
		self.attrNum = None 
		self.typeAttribute = None
		self.children = []
		self.leafOrNot = 0
		self.prediction = None
		self.probability = 0;
		self.splitAttribute = None;
		self.splitValue = None;
		self.nodes =[]
		self.leafNodes = []
		self.precision = 0.0
		self.parent = None
		self.searched = 0

attempts= int(sys.argv[4])
INSTANCE_LIMIT = attempts

def getEntropy(instance_set,verbose):

	#if continuous, need to split on something

	#if discrete, then figure out the total entropy

	#alternatives are just 1/0 for output
	#so just do -p_1log(p_1) - p_2log(p_2)
	if verbose:
		print '\t Entropy calculation: '
	positive=0.0
	negative=0.0
	for instance in instance_set:
		if verbose:
			print '\t',instance[-1]
		if (instance[-1])=='1':
			positive+=1
		else:
			if(instance[-1])=='0':
				negative+=1
	if verbose:
		print '\t pos: ',positive
		print '\t neg: ',negative
	if positive==0 or negative==0:
		return 0
	ppos = float(positive)/len(instance_set)
	pneg = float(negative)/len(instance_set)

	entropy = -1*ppos*float(math.log(ppos,2)) - pneg*float(math.log(pneg,2))
	
	if verbose:
		print "\t Entropy is ",entropy

	return entropy


def infoGain(instances,typeAttr,attrIndex,verbose):

	if verbose:
		print attrIndex

	totalGain = 0.0
	possibilities = []
	if typeAttr==' nominal':
		#need to get all the possibilities 

		#and also separate the instances into each possibility
		split_instances = {}
		for instance in instances:
			#if that value hasn't been recorded, add it as a possibility
			if not(instance[attrIndex] in possibilities):
				split_instances[instance[attrIndex]]=[instance]
				possibilities.append(instance[attrIndex])
			#and add whatever instance to the stuff. 
			else:
				split_instances[instance[attrIndex]].append(instance)

			
	if typeAttr==' numeric' or typeAttr=='numeric':
		#can do <= or >
		split_instances = {}
		split_instances['less_eq']=[]
		split_instances['greater']=[]
		#get the average and split down or up
		sumInstances = float(0.0)
		for instance in instances:
			if instance[attrIndex]=='?':
				instance[attrIndex]=0
			sumInstances+=float(instance[attrIndex])
		average = sumInstances / float(len(instances))
		if verbose:
			print 'average: ',average

		#now i know the average
		for instance in instances:
			if (float(instance[attrIndex])<=average):
				split_instances['less_eq'].append(instance)
			else:
				split_instances['greater'].append(instance)

		possibilities = ['less_eq','greater']

	entropies = {}
	for p in possibilities:
			entropies[p]=float(getEntropy(split_instances[p],verbose))
			numInstances = len(split_instances[p])
			weight = float(numInstances)/float(len(instances))
			totalGain = totalGain + float(entropies[p])*float(weight)
			if verbose:
				print p,numInstances,weight,entropies[p]

	totalGain = totalGain - float(getEntropy(instances,verbose))

	if verbose:
		print 'entropy before: ',float(getEntropy(instances,verbose))
	return totalGain


def ID3Recursive(tree,instancesLeft,attributes_1,typeAttribute_1,attrNum_1,recursive,splitAttribute,splitValue,parent):

	#print recursive
	recursive +=1
	tabs = '\t'*recursive


	attributes = deepcopy(attributes_1)
	typeAttribute = deepcopy(typeAttribute_1)
	attrNum = deepcopy(attrNum_1)

	current_node = TreeNode()
	current_node.data=instancesLeft
	current_node.attributes = attributes
	current_node.attrNum = attrNum 
	current_node.typeAttribute = typeAttribute
	current_node.children = []
	current_node.leafOrNot = 0
	current_node.prediction = None
	current_node.probability = 0
	current_node.parent = parent

	current_node.splitValue = splitValue

	positive = 0
	negative = 0

	for inst in instancesLeft:
		if inst[-1]=='1':
			positive+=1
		#print 'LEAF NODE: PREDICTION: ',float(positive)/float(len(instancesLeft))
	current_node.prediction = round(float(positive) / float(len(instancesLeft)))
	current_node.probability = float(positive)/float(len(instancesLeft))
	if current_node.probability >= 0.5:
		current_node.prediction = 1
	else:
		current_node.prediction = 0
	if current_node.prediction == 0:
		current_node.probability = float(1) - float(current_node.probability)


	if current_node.probability>=0.9:
		current_node.leafOrNot = 1
		tree.leafNodes.append(current_node)
		tree.nodes.append(current_node)
	#	print tabs,'leaf node, prediction: ',current_node.prediction
		return current_node


	if (len(instancesLeft)<=INSTANCE_LIMIT) or (len(attrNum)<=1) :
		current_node.leafOrNot = 1
		tree.leafNodes.append(current_node)
		tree.nodes.append(current_node)
	#	print tabs,"leaf node, prediction: ",current_node.prediction
		return current_node

	positive=0
	negative=0


	informationGain = {}
	#print attributes
	maxGain = float(0.0)
	bestattr=0
	#print typeAttribute
	#for key in attrNum:
	#		print key#
	#	print attributes


	for attr in attributes:
		# print attr
		informationGain[attr] = abs(infoGain(instancesLeft,typeAttribute[attrNum[attr]],attrNum[attr],0))
		if informationGain[attr] > maxGain:
			bestattr = attr
			maxGain = float(informationGain[attr])

	# if information gain not enough, return that node
	#print maxGain
	# if maxGain < 0.000000001:
	# #	print 'not enough information gain'
	# 	return None 


	if not bestattr:
		print 'ERROR - BUG DETECTED !'
		print attributes
		for instance in instancesLeft:
			print instance

		for attr in attributes:
			print attr
			informationGain[attr] = abs(infoGain(instancesLeft,typeAttribute[attrNum[attr]],attrNum[attr],1))
			print informationGain[attr]
		tree.nodes.append(current_node)
		print 'CANNOT FIND A BEST ATTRIBUTE'
		return current_node


	#print typeAttribute
	if typeAttribute[attrNum[bestattr]]==' nominal':
		#do a split on a nominal. so on each possible attribute
		#and each to the children of the current node

		#get all possibilities of the current attribute
		
		#print tabs,'split on ',bestattr,
		typeIsNominal = 1
		positive=0
		for instance in instancesLeft:
			if instance[-1]=='1':
				positive+=1
			else:
				negative+=1
		if ((positive==0) or (negative==0)):
			print ' xxxxxxxxx - PERFECT xxxxxxx \t\t\t',
		#print "   RESULT: ",positive,' positive vs ',negative,'negative and ',len(attributes)-1,' attributes left'
		
		possibilities = []
		split_instances = {}
		for instance in instancesLeft:
			#if that value hasn't been recorded, add it as a possibility
			if not(instance[attrNum[bestattr]] in possibilities):
				split_instances[instance[attrNum[bestattr]]]=[instance]
				possibilities.append(instance[attrNum[bestattr]])
			#and add whatever instance to the stuff. 
			else:
				split_instances[instance[attrNum[bestattr]]].append(instance)
			#print attrNum
			#print instance

	else:
	#attribute is numeric, so split on <= or > than. and keep track on number of splits 
		#print tabs,'split on ',bestattr,
		typeIsNominal = 0
		for instance in instancesLeft:
			if instance[-1]=='1':
				positive+=1
			else:
				negative+=1
		if ((positive==0) or (negative==0)):
			print ' xxxxxxxxx - PERFECT xxxxxxx \t\t\t',
		#print "   RESULT: ",positive,' positive vs ',negative,'negative and ',len(attributes)-1,' attributes left'
		
		split_instances = {}
		split_instances['less_eq']=[]
		split_instances['greater']=[]
		#get the average and split down or up
		sumInstances = float(0.0)
		for instance in instancesLeft:
			if instance[attrNum[bestattr]]=='?':
				instance[attrNum[bestattr]]=0
			sumInstances+=float(instance[attrNum[bestattr]])
		average = sumInstances / float(len(instancesLeft))
		# print average

		#now i know the average
		for instance in instancesLeft:
			if (float(instance[attrNum[bestattr]])<=average):
				split_instances['less_eq'].append(instance)
			else:
				split_instances['greater'].append(instance)

		possibilities = ['less_eq','greater']

	#figure out the number 
	index = int(attrNum[bestattr])
	typeAttribute.pop(index)
	attrNum.pop(bestattr)
	for key in attrNum:
		if attrNum[key]>index:
			attrNum[key]-=1

	newAttributes = [x for x in attributes if x!=bestattr]
	for key in split_instances:
		#print len(split_instances[key])
		#print 'splits'
		#print split_instances[key]
		pos=0.0
		for instance in split_instances[key]:
			instance.pop(index)
			if instance[-1]=='1':
				pos+=1
		prediction = float(pos)/float(len(split_instances[key]))
		#print tabs,'splitting on ',bestattr,' on value ',key, 'with ',len(split_instances[key]), 'instances '


		if typeIsNominal==1:
			newNode = ID3Recursive(tree,split_instances[key],newAttributes,typeAttribute,attrNum,recursive,bestattr,key,current_node)
		else:
			newNode = ID3Recursive(tree,split_instances[key],newAttributes,typeAttribute,attrNum,recursive,bestattr,average,current_node)
		current_node.children.append(newNode)

	current_node.splitAttribute = bestattr


	tree.nodes.append(current_node)
	return current_node	


def CreateTree(instances,Attribute_dict,attributes,typeAttribute,numberAttributes,attrNum):

	tree = ID3Tree()
	tree.data = instances
	tree.attr_dict = Attribute_dict
	tree.attributeList = attributes 
	tree.attrType = typeAttribute
	tree.attrNum = attrNum
	tree.attributeNumbers = numberAttributes
	tree.nodes=[]
	tree.leafNodes = []

	rootNode = TreeNode()
	rootNode = ID3Recursive(tree,instances,attributes,typeAttribute,attrNum,0,None,None,None)

	tree.root = rootNode

	#print 'Created tree has ',len(tree.nodes),' nodes and ',len(tree.leafNodes),' leaf nodes'
	return tree


def testInstance(testCase , Node ,attrDict,attrTypes,verbose):


	if Node.leafOrNot==1:
		if verbose:
			print 'leaf node  with test',testCase[-1],' and prediction ',Node.prediction
		if not testCase[-1]=='?':
			if int(testCase[-1])==int(Node.prediction):
			#	print 'correct!'
				return 1
			else:
			#	print 'incorrect!'
				return 0
		else:
			return 0


	splitIndex = attrDict[Node.splitAttribute]
	
	if verbose:
		print 'splitting down to ',Node.splitAttribute,' seeking value ',testCase[splitIndex]
	#otherwise, try to do the split by finding the correct child to split to 
	if attrTypes[splitIndex]==' nominal':
	#note that for numeric and nominal it's different
		for child in Node.children:
			if verbose:
				print child.splitValue,' vs ',testCase[splitIndex]
			if child.splitValue == testCase[splitIndex]:
				return testInstance(testCase,child,attrDict,attrTypes,verbose)
	else:
		if verbose:
			print Node.children[0].splitValue,' vs ',testCase[splitIndex]
		if testCase[splitIndex]=='?':
			return 0
		if Node.children[0].splitValue > float(testCase[splitIndex]):
			if verbose:
				print 'select less than or equal'
			return testInstance(testCase, Node.children[0],attrDict,attrTypes,verbose)
		else:
			if verbose:
				print 'select greater than '
			return testInstance(testCase, Node.children[1],attrDict,attrTypes,verbose)

		#it's numeric. so work in less than/greater than

	#in theory doesn't get here.. but in case the attribute doesn't exist:
	return 0


def testTree(testInstances,ID3Tree):

	#for each instance
	#get to the leaf and check against the last value
	#if correct, add to total
	#if incorrect, do not add
	correct = 0
	testInstances = testInstances[1:]
	for testCase in testInstances:
		#print '-----'
		correct += testInstance(testCase,ID3Tree.root,ID3Tree.attrNum,ID3Tree.attrType,0)

	accuracy = float(correct) / float(len(testInstances))

	print "Percentage accuracy: ",accuracy
	ID3Tree.precision = accuracy

	totalPositive=0
	for n in ID3Tree.leafNodes:
		totalPositive += n.prediction

	# print float(totalPositive)/len(ID3Tree.leafNodes)

	return ID3Tree


def ID3Stuff(trainFile,testFile,pruning,validateOrNot,alternative_training_data):

	#read attributes
	# if not(sys.argv==5):
	# 	print "Please input 5 arguments (script, trainFile, testFile, validationFile, verbose)"
	# else:
	fileName = sys.argv[1]
	file = open(fileName,'r')
	lines_array = []
	print "Reading file..."
	for line in file:
		lines_array.append((line.strip()).split(','))

	#first line has the attribute names
	attributes = lines_array[0]
	attributes = attributes[0:-1]
	attrNum = {}

	#keep a record of in what position everything is
	k=0
	for attr in attributes:
		attrNum[attr]=k
		k+=1

	numberAttributes = len(attributes)-1
	#second line has the type of attribute 
	typeAttribute = lines_array[1]

	#now capture examples for all attributes
	if alternative_training_data==None:
		instances = lines_array[2:] 
	else:
		instances = alternative_training_data
		
		
	print "Number of instances: ",len(instances)
	print "Number of attributes: ",len(instances[1])

	
	print "Fixing to numbers if needed..."

	for instance in instances:
		if instance[-1]=='?':
			flag=0
			for i in instances:
				if (instance[:-1]==i[:-1]) and (i[-1] != '?'):
					instance[-1]=i[-1]
					flag=1
				break
			if flag==0:
				instance[-1]='0'

	Attribute_dict = {}
	m=0
	for A in attributes :
		data = []
		for instance in instances:
			data.append(instance[m])
		Attribute_dict[A] = data
		print A ,':   \t', len(Attribute_dict[A])
		m+=1

	print "Done fixing, now create trees..."
	file.close()

	tree = CreateTree(instances,Attribute_dict,attributes,typeAttribute,numberAttributes,attrNum)
	
	###################################################################################################
	if validateOrNot==0:
		return tree

	print " CREATED TREE - TESTING..."

	testFileName = sys.argv[2]
	file = open(testFileName,'r')
	lines_array = []
	print "Reading file..."
	for line in file:
		lines_array.append((line.strip()).split(','))

	#first line has the attribute names
	attributes = lines_array[0]
	attributes = attributes[0:-1]
	attrNum = {}
	instances = lines_array[2:] 

	#keep a record of in what position everything is
	k=0
	for attr in attributes:
		attrNum[attr]=k
		k+=1

	numberAttributes = len(attributes)-1
	#second line has the type of attribute 
	typeAttribute = lines_array[1]

	positive = 0.0
	testAverages = {}
	l=0
	for attr in attributes:
#		print "average for : ",attr,
		testAverages[attr] = getAvg(instances,l)
		if typeAttribute[l]==' nominal':
			testAverages[attr] = int(round(testAverages[attr]))
#		print averages[attr]
		l+=1
	
	print "Fixing to numbers if needed..."

	positive = 0.0
	positive = 0.0
	for instance in instances:
		i=0
		for attr in instance[:-1]:
			if attr=='?':
				attr = testAverages[attributes[i]]
			i+=1

		if instance[-1]=='1':
			positive+=1



	print 'percentage of 1: ',float(positive)/float(len(instances))
	print "Done loading data, now testing the testing set ..."
	file.close()

	newTree = testTree(instances,tree)
	continuePrune = pruning
	while continuePrune == 1:
		[continuePrune,newTree] = prune(instances,newTree)
		print 'Current precision after prune: ' ,newTree.precision
	return newTree

# precision = []
# treeSize = []
# leafSize = []
# percentageLeaves = []
# xAxis = range(10000,int(sys.argv[3]),500)
# for i in xAxis:
# 	INSTANCE_LIMIT = i
# 	temp_tree = ID3Stuff(trainFile,testfile)
# 	precision.append(temp_tree.precision)
# 	treeSize.append(len(temp_tree.nodes))
# 	leafSize.append(len(temp_tree.leafNodes))
# 	percentageLeaves.append(float(len(temp_tree.leafNodes))/float(len(temp_tree.nodes)))

# plt.figure(1)
# plt.subplot(311)
# plt.plot(xAxis,precision)
# plt.ylabel('Precision')
# plt.grid(True)

# plt.subplot(312)
# plt.plot(xAxis,treeSize,xAxis,leafSize)
# plt.ylabel('Tree Size')
# plt.grid(True)

# plt.subplot(313)
# plt.plot(xAxis,percentageLeaves)
# plt.ylabel('Percentage leaves of the tree')
# plt.grid(True)

# plt.show()

def getAvg(instances,index):
	
	sum = 0
	length = 0
	for instance in instances:
		if instance[index] != '?':
			sum += float(instance[index])
			length+=1
	
	return float(sum) / float(length)
		
		
def prune(testInstances,tree):

	print 'initial precision: ',tree.precision,'with ',len(tree.leafNodes),' leaf nodes out of ',len(tree.nodes)

	bestPrecision = deepcopy(tree.precision)
	treeCopy= (tree)
	improvedFlag=0
    
	unsearchedCount=0
	for leafNode in treeCopy.leafNodes:
		if leafNode.searched == 1:
			unsearchedCount=unsearchedCount+1
	print "Number of node searched:",unsearchedCount
    
	for leafNode in treeCopy.leafNodes:
	    #if the parent is not a leaf
	    if leafNode.parent.leafOrNot == 0 and leafNode.parent.searched == 0:
	        #perform a new test with that one set as a leaf
	        testNode = leafNode.parent
	        testNode.leafOrNot = 1
	        treeCopy = testTree(testInstances, treeCopy)
	        testNode.searched=1
	        
	        if treeCopy.precision <= bestPrecision:
	            #then reset that node to 0
	            testNode.leafOrNot = 0
	        else:
	            #set recursively that node and all nodes to leaf nodes
	            setLeaf(testNode, treeCopy.leafNodes)
	            bestPrecision= treeCopy.precision
	            improvedFlag = 1
	            return [improvedFlag, treeCopy]
				
				
	     
	print 'improvedFlag = ',improvedFlag,' down to ',bestPrecision,' (',tree.precision,')'
	print 'leaf nodes: ',len(tree.leafNodes),' out of ',len(tree.nodes),'nodes'

	
	tree.precision = bestPrecision

	return [improvedFlag,tree]

def setLeaf(someNode, leaves):

    someNode.leafOrNot = 1
    someNode.searched = 1
    if someNode not in leaves:
        leaves.append(someNode)
    for child in someNode.children:
        setLeaf(child,leaves)
        
    return 
    
def PrintTree(tree):
	root = tree.root
	printRecursive(root,0)
	
def printRecursive(node,recursionLevel):
	
	recursionLevel = recursionLevel+1
	tabs = '\t'*recursionLevel
	if node.leafOrNot==1:
		print tabs,'Attribute : ',node.parent.splitAttribute,' value: ',node.splitValue,' LEAF NODE: Prediction: ',node.prediction
		return
	
	else:
		if node.parent==None:
			print 'Root Node, split on : ',node.splitAttribute
		else:
			print tabs,'Attribute: ',node.parent.splitAttribute,' value: ',node.splitValue,' Move down to ',node.splitAttribute

	for kid in node.children:
		printRecursive(kid,recursionLevel)
		
	return

def getData(train_file,size_portion):
	
	
	file = open(train_file,'r')
	lines_array = []
	print "Reading file..."
	for line in file:
		lines_array.append((line.strip()).split(','))

	#first line has the attribute names
	attributes = lines_array[0]
	attributes = attributes[0:-1]
	attrNum = {}

	#keep a record of in what position everything is
	k=0
	for attr in attributes:
		attrNum[attr]=k
		k+=1

	numberAttributes = len(attributes)-1
	#second line has the type of attribute 
	typeAttribute = lines_array[1]

	#now capture examples for all attributes
	instances = lines_array[2:] 
	
	print 'getting it all together and requested size is: ',len(instances),' and we ask for ',size_portion

	for instance in instances:
		if instance[-1]=='?':
			flag=0
			for i in instances:
				if (instance[:-1]==i[:-1]) and (i[-1] != '?'):
					instance[-1]=i[-1]
					flag=1
				break
			if flag==0:
				instance[-1]='0'
	file.close()
	
	
	# newInstances = []
	# print 'now select randomly from the data set'
	# for i in range(size_portion):
	# #	print 'instances: ',len(instances)
	# 	chosen_index = random.randrange(1,len(instances))
	# 	temp = instances[chosen_index]
	# 	if i%100==0:
	# 		print 'added ',i,' out of ',size_portion
	# 	while temp in newInstances:
	# 		chosen_index = random.randrange(len(instances))
	# 		temp = instances[chosen_index]
			
	# 	newInstances.append(temp)
		
	print 'shuffling and selecting instances'
	random.shuffle(instances)
	return instances[1:size_portion]
	#return newInstances
			
def PerformTest(tree,testFile):
	
	print " CREATING TEST FILE OUTPUT "
	file = open(testFile,'r')
	lines_array = []
	print "Reading file..."
	for line in file:
		lines_array.append((line.strip()).split(','))

	#first line has the attribute names
	attributes = lines_array[0]
	attributes = attributes[0:-1]
	attrNum = {}
	instances = lines_array[2:] 

	#keep a record of in what position everything is
	k=0
	for attr in attributes:
		attrNum[attr]=k
		k+=1

	numberAttributes = len(attributes)-1
	#second line has the type of attribute 
	typeAttribute = lines_array[1]

	positive = 0.0
	testAverages = {}
	l=0
	for attr in attributes:
#		print "average for : ",attr,
		testAverages[attr] = getAvg(instances,l)
		if typeAttribute[l]==' nominal':
			testAverages[attr] = int(round(testAverages[attr]))
#		print averages[attr]
		l+=1
	
	print "Fixing Unknowns..."

	positive = 0.0
	for instance in instances:
		i=0
		for attr in instance[:-1]:
			if attr=='?':
				attr = testAverages[attributes[i]]
			i+=1

		if instance[-1]=='1':
			positive+=1

	print "Done loading data, now predicting the unknown test set....."
	file.close()
	
	out = csv.writer(open("output.csv", "w"), delimiter=',', quoting=csv.QUOTE_ALL)
	out.writerow(attributes)
	out.writerow(lines_array[1])
	
	instances = instances[1:]
	for testCase in instances:
		prediction = valuePredictor(testCase,tree.root,tree.attrNum,tree.attrType)
		#print prediction
		testCase[-1] = prediction
		out.writerow(testCase)

	return 

	# done with function


def predictor (testCase , Node ,attrDict,attrTypes):


	if Node.leafOrNot==1:
		if not testCase[-1]=='?':
			if int(testCase[-1])==int(Node.prediction):
				return 1
			else:
				return 0
		else:
			return 0


	splitIndex = attrDict[Node.splitAttribute]
	
	#otherwise, try to do the split by finding the correct child to split to 
	if attrTypes[splitIndex]==' nominal':
	#note that for numeric and nominal it's different
		for child in Node.children:
			if child.splitValue == testCase[splitIndex]:
				return predictor(testCase,child,attrDict,attrTypes)
	else:
		if testCase[splitIndex]=='?':
			return 0
		if Node.children[0].splitValue > float(testCase[splitIndex]):
			return predictor(testCase, Node.children[0],attrDict,attrTypes)
		else:
			return predictor(testCase, Node.children[1],attrDict,attrTypes)

		#it's numeric. so work in less than/greater than

	#in theory doesn't get here.. but in case the attribute doesn't exist:
	return 0
	

def valuePredictor (testCase , Node ,attrDict,attrTypes):


	if Node.leafOrNot==1:
		return Node.prediction


	splitIndex = attrDict[Node.splitAttribute]
	
	#otherwise, try to do the split by finding the correct child to split to 
	if attrTypes[splitIndex]==' nominal':
	#note that for numeric and nominal it's different
		for child in Node.children:
			if child.splitValue == testCase[splitIndex]:
				return predictor(testCase,child,attrDict,attrTypes)
	else:
		if testCase[splitIndex]=='?':
			return 0
		if Node.children[0].splitValue > float(testCase[splitIndex]):
			return valuePredictor(testCase, Node.children[0],attrDict,attrTypes)
		else:
			return valuePredictor(testCase, Node.children[1],attrDict,attrTypes)

		#it's numeric. so work in less than/greater than

	#in theory doesn't get here.. but in case the attribute doesn't exist:
	return 0

########################### MAIN FUNCTION HERE



model = int(sys.argv[5])
test_file = sys.argv[3]
validation_file = sys.argv[2]
train_file = sys.argv[1]

pruning = int(sys.argv[6])

if model==1:
	#perform a simple tree and print
	tree = ID3Stuff(train_file,validation_file,0,0,None)
if model==2:
	#perform a tree and test on validation set with/without pruning
	tree = ID3Stuff(train_file,validation_file,0,1,None)
	PrintTree(tree)
	prunedTree = ID3Stuff(train_file,validation_file,1,1,None)
	PrintTree(prunedTree)
	
	print 'Accuracy before: ',tree.precision,' vs accuracy after pruning: ',prunedTree.precision
if model==3:
	#perform a test and print out on other file
	tree = ID3Stuff(train_file,validation_file,pruning,1,None)
	PrintTree(tree)
	PerformTest(tree,test_file)
if model==4:
	#show different validation sets for a different number of training size (1/10 each until 1)
	precision = []
	x = []
	for i in range(1,26):
		num_samples = (49982*i)/25
		newTrainingData = getData(train_file,num_samples)
		tree = ID3Stuff(train_file,validation_file,pruning,1,newTrainingData)
		precision.append(tree.precision)
		x.append(float(i)/float(25))
		
	#plot this
	fig = plt.figure()
	plt.plot(x,precision)
	plt.xlabel('Percentage of data set used for tree')
	plt.ylabel('Validation set accuracy')
	if pruning:
		titleString = 'Learning curve for pruned trees'
	else:
		titleString = 'Learning curve for unpruned trees'
	fig.suptitle(titleString)
	plt.grid(True)
	plt.show()
	
	
	