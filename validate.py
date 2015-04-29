def validate(tree):

	#go through the validation data set and find the prediction accuracy
	#first we need a prediction in each of the leafs in the tree

	#read attributes from validation file

	# if not(sys.argv==5):
	# 	print "Please input 5 arguments (script, trainFile, testFile, validationFile, verbose)"
	# else:
	fileName = 'bvalidate.csv'
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