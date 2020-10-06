import math

class Tree(object):
	def __init__(self):
		self.child = []
		self.label = -1
		self.attribute = -1
		self.nExamples = -1
	
	def print_1(self, i, Examples, featuresList, listAttributes, Target_Attribute, classNames, output):
		s = '|   '*i
		if self.label != -1:
			print(': ', end = '', file = output)
			print(classNames[self.label], end = '', file = output)
		else:
			print('', file = output)
			j = 0
			for c in self.child:
				print(s, end = '', file = output)
				print(featuresList[self.attribute], end = ' = ', file = output)
				print(listAttributes[self.attribute].values[j], end = '', file = output)
				c.print_1(i + 1, Examples, featuresList, listAttributes, Target_Attribute, classNames, output)
				j += 1

	def print(self, i, Examples, featuresList, listAttributes, Target_Attribute, output):
		s = '|   '*i
		if self.label != -1:
			print(': ', end = '', file = output)
			print(self.label, sep = ' ', file = output)
		else:
			print('', file = output)
			j = 0
			for c in self.child:
				print(s, end = '', file = output)
				print(featuresList[self.attribute], end = ' = ', file = output)
				print(listAttributes[self.attribute].values[j], end = '', file = output)
				c.print(i + 1, Examples, featuresList, listAttributes, Target_Attribute, output)
				j += 1

	def countLeaves(self):
		if self.label != -1:
			return 1
		else:
			nLeaves = 0
			for c in self.child:
				nLeaves += c.countLeaves()
			return nLeaves

	def size(self):
		if self.label != -1:
			return 1
		else:
			size = 1
			for c in self.child:
				size += c.size()
			return size

class Attribute(object):
	def __init__(self):
		self.index = None
		self.values = []	

#Read input from file
def readInput(inputFileName):
	Examples = []
	fin = open(inputFileName, "r")
	featuresList = fin.readline().strip('\r\n').split(',')
	# subtract name and targetAttribute
	nAttributes = len(featuresList)
	i = 0
	for line in fin:
		line = line.strip('\n')
		tmp = line.split(',')
		example = [i, tmp[0]]
		for j in range (1, nAttributes + 2):
			example.append(tmp[j])
		Examples.append(example)
		i += 1
	
	# i is now number of examples
	targetAttribute = findTargetAttribute(Examples)
	targetAttribute.sort()
	listAttributes = createListAttribute(Examples, list(range(0, i)), nAttributes)
	return [Examples, featuresList, listAttributes, targetAttribute]

#Find list of value of TargetAttrubute 
def findTargetAttribute(Examples):
	nExamples = len(Examples)
	setOfClasses = set()
	for i in range(nExamples):
		setOfClasses.add(Examples[i][-1])
	return list(setOfClasses)

#Create list of attributes 
def createListAttribute(Examples, listUseExamples, nAttributes):
	listAttributes = []
	for i in range(nAttributes):
		attr = Attribute()
		attr.index = i
		attrValues = set()
		for j in listUseExamples:
			attrValues.add(Examples[j][i + 2])
		attr.values = list(attrValues)
		listAttributes.append(attr)
	return listAttributes

#Return list count frequency of values in TargetAttribute
def countTargetAttribute(Examples, listUseExamples, Target_Attribute):
	nExamplesAtTarget = []
	for i in range(len(Target_Attribute)):
		nExamplesAtTarget.append(0)
	for i in listUseExamples:
		nExamplesAtTarget[Target_Attribute.index(Examples[i][-1])] += 1
	return nExamplesAtTarget

#Find most common value in a list of integers
def findMostCommonValue(listNum):
	maxList = max(listNum)
	return [listNum.index(maxList), maxList]

#Create a decision tree
def decisionTree(Examples, listUseExamples, Target_Attribute, listAttributes, iAttribute):
	decisionTreeRoot = Tree()
	decisionTreeRoot.attribute = iAttribute
	attr = listAttributes[iAttribute]
	for i in range(len(attr.values)):
		decisionTreeRoot.child.append([])
	for i in listUseExamples:
		value_index = attr.values.index(Examples[i][iAttribute + 2])
		decisionTreeRoot.child[value_index].append(Examples[i][0])
	return decisionTreeRoot

#Return list of examples whose attribute[iAttribute] equal v
def listExamplesV(Examples, listUseExamples, listAttributes, iAttribute):
	attr = listAttributes[iAttribute]
	ListExamplesV = []
	for i in range(len(attr.values)):
		ListExamplesV.append([])
	for i in listUseExamples:
		ListExamplesV[attr.values.index(Examples[i][iAttribute + 2])].append(i)
	return ListExamplesV

#Calculate entropy
def calcEntropy(Examples, listUseExamples, listAttributes, listUseAttributes, Target_Attribute):
	listEntropies = []
	totalInTree = len(listUseExamples)
	for i in listUseAttributes:
		attr = listAttributes[i]
		ListExamplesV = listExamplesV(Examples, listUseExamples, listAttributes, i)
		entropy = 0.0
		for j in range(len(ListExamplesV)):
			totalInBranch = len(ListExamplesV[j])
			countEachBranch = countTargetAttribute(Examples, ListExamplesV[j], Target_Attribute)
			for cnt in countEachBranch:
				if cnt != 0:
					entropy += (-cnt)*math.log(cnt/totalInBranch, 2)
		entropy /= 	totalInTree
		listEntropies.append(entropy)
	return listEntropies

# return index in listAttributes of the best classifed attribute
def findBestAttribute(Examples, listUseExamples, listAttributes, listUseAttributes, Target_Attribute):
	listEntropies = calcEntropy(Examples, listUseExamples, listAttributes, listUseAttributes, Target_Attribute)
	return listUseAttributes[listEntropies.index(min(listEntropies))]

# return index of Target_Attribute that all examples are same
def checkAllSameValues(Examples, listUseExamples, Target_Attribute):
	nExamplesAtTarget = countTargetAttribute(Examples, listUseExamples, Target_Attribute)
	if nExamplesAtTarget.count(0) != len(Target_Attribute) - 1:
		return -1
	else:
		maxN = max(nExamplesAtTarget)
		return [nExamplesAtTarget.index(maxN), maxN]

#ID3 algorithm
def ID3(Examples, listUseExamples, listAttributes, listUseAttributes, Target_Attribute):
	root = Tree()
	if len(listUseAttributes) == 0:
		[root.label, root.nExamples] = findMostCommonValue(countTargetAttribute(Examples, listUseExamples, Target_Attribute))
	else:
		allSameValues = checkAllSameValues(Examples, listUseExamples, Target_Attribute)
		if (allSameValues != -1):
			[root.label, root.nExamples] = allSameValues
		else:
			bestClassifiedAttribute = findBestAttribute(Examples, listUseExamples, listAttributes, listUseAttributes, Target_Attribute)
			root.attribute = bestClassifiedAttribute
			ListExamplesV = listExamplesV(Examples, listUseExamples, listAttributes, bestClassifiedAttribute)
			for i in range(len(ListExamplesV)):
				subTree = Tree()
				if ListExamplesV[i] == []:
					[subTree.label, subTree.nExamples] = findMostCommonValue(countTargetAttribute(Examples, listUseExamples, Target_Attribute))
				else:
					newListUseAttributes = listUseAttributes[:]
					newListUseAttributes.remove(bestClassifiedAttribute)
					subTree = ID3(Examples, ListExamplesV[i], listAttributes, newListUseAttributes, Target_Attribute)
				root.child.append(subTree)
	return root

# main
def main():
	[Examples, nExamples, listAttributes, nAttributes, Target_Attribute, nTargetAttributes] = readInput("zoo.data")
	featuresList = ['hair', 'feathers', 'eggs', 'milk', 'airbone', 'aquatic', 'predator', 'toothed', 'backbone', 'breathes', 'venomous', 'fin', 'legs', 'tail', 'domestic', 'catsize']
	classNames = ['mammal', 'bird', 'reptile', 'aquatic', 'amphibian', 'insect', 'other']

	id3Tree = ID3(Examples, list(range(0, nExamples)), listAttributes, list(range(0, nAttributes)), Target_Attribute)
	id3Tree.print_1(0, Examples, featuresList, listAttributes, Target_Attribute, classNames)
	id3Tree.print(0, Examples, featuresList, listAttributes, Target_Attribute)

if __name__ == '__main__':
	main()

