import sys
import math


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
		self.attribute = None
		self.data = None
		self.children = None



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



def infoGain(instances,typeAttr,attrNum):

	totalGain = 0.0
	possibilities = []
	if typeAttr==' nominal':
		#need to get all the possibilities 

		#and also separate the instances into each possibility
		split_instances = {}
		for instance in instances:
			#if that value hasn't been recorded, add it as a possibility
			if not(instance[attrNum] in possibilities):
				split_instances[instance[attrNum]]=[instance]
				possibilities.append(instance[attrNum])
			#and add whatever instance to the stuff. 
			else:
				split_instances[instance[attrNum]].append(instance)
			#print attrNum
			#print instance
			
	if typeAttr==' numeric' or typeAttr=='numeric':
		#can do <= or >
		split_instances = {}
		split_instances['less_eq']=[]
		split_instances['greater']=[]
		#get the average and split down or up
		sumInstances = float(0.0)
		for instance in instances:
			if instance[attrNum]=='?':
				instance[attrNum]=0
			sumInstances+=float(instance[attrNum])
		average = sumInstances / float(len(instances))
		# print average

		#now i know the average
		for instance in instances:
			if (instance[attrNum]<=average):
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

	

	

def ID3Recursive(tree,instancesLeft,attributes,typeAttribute,attrNum):

	cur_tree = ID3Tree()
	cur_tree = tree

	#save in a dictionary
	typeAttributes={}
	typeAttributesNum = {}
	i=0
	for attribute in attributes:
		typeAttributes[attribute]=typeAttribute[i]
		typeAttributesNum[attribute]=i
		i+=1

	#now get entropy for each
	entropy = {}
	entropy_now = 0.0
	#print 'hi there, im recursive '
	entropy_now = float(getEntropy(instancesLeft))
	print entropy_now

	informationGain = {}
	#print attributes
	for attr in attributes:
		# print attr
		informationGain[attr] = infoGain(instancesLeft,typeAttributes[attr],attrNum[attr])

	for attr in attributes:
		print 'attribute ',attr,' : ',informationGain[attr]


	#now check the largest information gain and select that to partition from




	# for attr in attributes:
	# 	entropy[attr] = getEntropy(attributes,instancesLeft,attr)



def CreateTree(instances,Attribute_dict,attributes,typeAttribute,numberAttributes,attrNum):

	tree = ID3Tree()
	tree.data = instances
	tree.attr_dict = Attribute_dict
	tree.attributeList = attributes 
	tree.attrType = typeAttribute
	tree.attributeNumbers = numberAttributes

	rootNode = TreeNode()
	rootNode = ID3Recursive(tree,instances,attributes,typeAttribute,attrNum)


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






