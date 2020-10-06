import id3
import training
import math
import random
import sys
import os

def solve(input, trainPercent, output = sys.stdout):
	random.seed()
	[Examples, featuresList, listAttributes, Target_Attribute] = id3.readInput(input)
	train = []
	nExamples = len(Examples)
	for i in range(nExamples):
		train.append(0)
	randArray = random.sample(range(0, nExamples), math.ceil(nExamples*trainPercent))
	for i in randArray:
		train[i] = 1
	TrainSet = []
	TestSet = []
	for i in range(nExamples):
		if train[i] == 1:
			TrainSet.append(Examples[i])
		else:
			TestSet.append(Examples[i])

	output.write('=== Run information ===\n\n')
	output.write('Relation:    %s\n' % input[:input.find('.')])
	output.write('Instances:   %d\n' % nExamples)
	output.write('Attributes:  %d\n' % (len(listAttributes) + 1))
	for attr in featuresList:
		output.write('             %s\n' % attr)
	output.write('             class\n')
	output.write('Test mode:split %0.2f%c train, reminder test\n\n' % (trainPercent*100, 37))
	
	output.write('=== Classifier model (full training set) ===\n\n')
	output.write('ID3 tree\n\n')
	
	[id3Tree, timeElapsed] = training.training(TrainSet, listAttributes, Target_Attribute)
	id3Tree.print(0, TrainSet, featuresList, listAttributes, Target_Attribute, output)
	print('Time taken to build model: %0.2f\n' % timeElapsed, file = output)
	
	output.write('=== Evaluation on test split ===\n')
	output.write('=== Summary ===\n\n')
	[ConfusionMatrix, wrongPredict] = training.confusionMatrix(TestSet, listAttributes, Target_Attribute, id3Tree)
	training.statistic(ConfusionMatrix, wrongPredict, Target_Attribute, output)
	training.printConfusionMatrix(ConfusionMatrix, Target_Attribute, output)
	print('Classified succeeded.')
	

if __name__ == '__main__':
	trainPercent = float(sys.argv[1])/100
	cwd = os.getcwd()
	for input in sys.argv[2:]:
		if (os.path.exists(cwd + '\\' + input)):
			outputFilename = input[:input.find('.')] + '.out'
			output = open(outputFilename, "w") 
			solve(input, trainPercent, output)
			output.close()
		else:
			print('Non exist file ' + input)
