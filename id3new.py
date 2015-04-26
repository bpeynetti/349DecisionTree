import sys
import math
from copy import deepcopy


class ID3Tree(object):
	def __init__(self):
		self.data = None
		self.attr_dict = None
		self.attributeList = None
		self.attrType = None
		self.root = None
		self.attributeNumbers = 0

class TreeNode(object):
	def __init__(self):
		self.attributes = None
		self.data = None
		self.children = None
		self.attrNum = None
		self.typeAttribute = None

def getEntropy(instance_set):

	#if continuous, need to split on something

	#if discrete, then figure out the total entropy

	#alternatives are just 1/0 for output
	#so just do -p_1log(p_1) - p_2log(p_2)

	positive=0.0
	negative=0.0
	for instance in instance_set:
		#print instance[-1]
		if (instance[-1])=='1':
			positive+=1
		else:
			if(instance[-1])=='0':
				negative+=1
	#print positive
	#print negative
	if positive==0 or negative==0:
		return 0
	ppos = float(positive)/len(instance_set)
	pneg = float(negative)/len(instance_set)

	entropy = -1*ppos*float(math.log(ppos,2)) - pneg*float(math.log(pneg,2))
	#print "Entropy is ",entropy

	return entropy


def infoGain(instances,typeAttr,attrIndex):

#	print attrIndex
#	print instances[0]

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
			#print attrIndex
			#print instance
			
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
		# print average

		#now i know the average
		for instance in instances:
			if (instance[attrIndex]<=average):
				split_instances['less_eq'].append(instance)
			else:
				split_instances['greater'].append(instance)

		possibilities = ['less_eq','greater']

	entropies = {}
	for p in possibilities:
			entropies[p]=float(getEntropy(split_instances[p]))
			numInstances = len(split_instances[p])
			weight = float(numInstances)/float(len(instances))
			totalGain = totalGain + float(entropies[p])*float(weight)
			# print numInstances,weight

	totalGain = totalGain - float(getEntropy(instances))

	return totalGain


def ID3Recursive(tree,instancesLeft,attributes_1,typeAttribute_1,attrNum_1,recursive):

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
	positive = 0
	negative = 0

	if len(instancesLeft)==1:
		for inst in instancesLeft:
			if inst[-1]=='1':
				positive+=1
		print 'LEAF NODE: PREDICTION: ',float(positive)/float(len(instancesLeft))
		return None


	#now get entropy for each
	entropy = {}
	entropy_now = 0.0
	#print 'hi there, im recursive '
	entropy_now = float(getEntropy(instancesLeft))
	#print entropy_now

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
		informationGain[attr] = abs(infoGain(instancesLeft,typeAttribute[attrNum[attr]],attrNum[attr]))
		if informationGain[attr] > maxGain:
			bestattr = attr
			maxGain = float(informationGain[attr])

	# if information gain not enough, return that node
	#print maxGain
	# if maxGain < 0.000000001:
	# #	print 'not enough information gain'
	# 	return None 


	if not bestattr:
		return current_node


	#print typeAttribute
	if typeAttribute[attrNum[bestattr]]==' nominal':
		#do a split on a nominal. so on each possible attribute
		#and each to the children of the current node

		#get all possibilities of the current attribute
		
		#print tabs,'split on ',bestattr,
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
			if (instance[attrNum[bestattr]]<=average):
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
		print tabs,'splitting on ',bestattr,' on value ',key, 'with ',len(split_instances[key]), 'instances '


		#print split_instances[key][0]
		##for k in attrNum:
	#		print k,' : ',attrNum[k],' , '
#		print ''	
		newNode = ID3Recursive(tree,split_instances[key],newAttributes,typeAttribute,attrNum,recursive)
		current_node.children.append(newNode)

	return current_node	


def CreateTree(instances,Attribute_dict,attributes,typeAttribute,numberAttributes,attrNum):

	tree = ID3Tree()
	tree.data = instances
	tree.attr_dict = Attribute_dict
	tree.attributeList = attributes 
	tree.attrType = typeAttribute
	tree.attributeNumbers = numberAttributes

	rootNode = TreeNode()
	rootNode = ID3Recursive(tree,instances,attributes,typeAttribute,attrNum,0)


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
instances = lines_array[2:] 
print "Number of instances: ",len(instances)
print "Number of attributes: ",len(instances[1])

print "Fixing to numbers if needed..."

for instance in instances:
	i=0
	for attr in instance:
		if typeAttribute[i]==' numeric':
			if attr=="?":
				attr=-1
			attr = float(attr)
		if typeAttribute[i]==' nominal':
			if attr=='?':
				attr=0
		i+=1

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






