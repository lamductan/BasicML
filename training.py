import id3
import math
import time

def dfs(test, decisionTree, listAttributes, Target_Attribute):
	if decisionTree.label != -1:
		return Target_Attribute[decisionTree.label]
	else:
		attrIndex = decisionTree.attribute
		attr = listAttributes[attrIndex]
		subTree = decisionTree.child[attr.values.index(test[attrIndex + 2])]
		return dfs(test, subTree, listAttributes, Target_Attribute)

def predictTarget(test, decisionTree, listAttributes, Target_Attribute):
	return dfs(test, decisionTree, listAttributes, Target_Attribute)

def training(TrainSet, listAttributes, Target_Attribute):
	startTime = time.time()
	id3Tree = id3.ID3(TrainSet, list(range(len(TrainSet))), listAttributes, list(range(0, len(listAttributes))), Target_Attribute)
	timeElapsed = time.time() - startTime
	return [id3Tree, timeElapsed]

def confusionMatrix(TestSet, listAttributes, Target_Attribute, id3Tree):
	size = len(Target_Attribute)
	ConfusionMatrix = []
	for i in range(size):
		row = []
		for j in range(size):
			row.append(0)
		ConfusionMatrix.append(row)

	CountTargetAttribute = id3.countTargetAttribute(TestSet, list(range(0, len(TestSet))), Target_Attribute)
	wrongPredict = []
	for i in range(len(TestSet)):
		predict = predictTarget(TestSet[i], id3Tree, listAttributes, Target_Attribute)
		if predict != TestSet[i][-1]:
			wrongPredict.append([i, predict])

	for i in range(len(wrongPredict)):
		testIndex = wrongPredict[i][0]
		predict = wrongPredict[i][1]
		ConfusionMatrix[Target_Attribute.index(TestSet[testIndex][-1])][Target_Attribute.index(predict)] += 1
	for i in range(size):
		ConfusionMatrix[i][i] = CountTargetAttribute[i] - sum(ConfusionMatrix[i])
	return [ConfusionMatrix, wrongPredict]

def printConfusionMatrix(ConfusionMatrix, Target_Attribute, output):
	print('\n\n=== Confusion Matrix ===', file = output)
	nTargetAttributes = len(Target_Attribute)
	for i in range(nTargetAttributes):
		print('      %c ' % (97+i), end = '', file = output)
	print('  <-- classified as', file = output)
	for i in range(len(ConfusionMatrix)):
		for data in ConfusionMatrix[i]:
			print('%7d ' % data, end = '', file = output)
		print('|  %c = %s' %(97+i, Target_Attribute[i]), file = output)
	output.write('\n')

def weightedAverage(a,b):
	size = len(a)
	WeightedAverage = 0
	for i in range(size):
		WeightedAverage += a[i]*b[i]
	return WeightedAverage/sum(b)

def statistic(ConfusionMatrix, wrongPredict, Target_Attribute, output):
	#ConfusionMatrix = [[41, 0, 0, 0, 0, 0, 0], [0, 3, 0, 0, 0, 1, 1], [0, 0, 20, 0, 0, 0, 0], [0, 0, 0, 5, 0, 3, 0], [0, 1, 0, 0, 3, 0, 0], [0, 0, 0, 1, 0, 9, 0], [0, 0, 0, 0, 0, 0, 13]]
	size = len(ConfusionMatrix)
	sumRows = []
	for i in range(size):
		sumRows.append(sum(ConfusionMatrix[i]))
	sumCols = []
	for i in range(size):
		s = 0
		for j in range(size):
			s += ConfusionMatrix[j][i]
		sumCols.append(s)

	numOfInstances = sum(sumRows)
	IncorrectClassifedInstances = len(wrongPredict)
	CorrectClassifedInstances = numOfInstances - IncorrectClassifedInstances
	IncorrectPercentage = IncorrectClassifedInstances / numOfInstances*100
	CorrectPercentage = 100 - IncorrectPercentage

	sumDiag = 0
	for i in range(size):
		sumDiag += ConfusionMatrix[i][i]
	Ne = 0
	for i in range(size):
		Ne += sumRows[i]*sumCols[i]
	Ne /= numOfInstances
	kappa = (sumDiag - Ne) / (numOfInstances - Ne)
	output.write('Correctly Classified Instances           %5d         %0.4f %c\n' %(CorrectClassifedInstances, CorrectPercentage, 37))
	output.write('Incorrectly Classified Instances         %5d         %0.4f %c\n' %(IncorrectClassifedInstances, IncorrectPercentage, 37))
	output.write('Kappa statistic                          %0.4f\n' % kappa)
	output.write('Total numbers of Instances               %5d\n\n' % numOfInstances)


	TPRate = []
	FPRate = []
	Recall = []
	Precision = []
	F1Score = []
	MCC = []
	for i in range(size):
		if sumCols[i] != 0 and sumRows[i] != 0:
			TP = ConfusionMatrix[i][i]
			FP = sumCols[i] - TP
			TN = numOfInstances - sumRows[i] - FP
			FN = sumRows[i] - TP

			TPRate.append(TP/(TP+FN))
			FPRate.append(FP/(FP+TN))
			Precision.append(TP/(TP+FP))
			Recall.append(TP/(TP+FN))
			if Precision[i]*Recall[i] != 0:
				F1Score.append(2 / (1/Precision[-1] + 1/Recall[-1]))
			else:
				F1Score = 0
			MCC.append((TP*TN-FP*FN)/ math.pow((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN),1/2))
		else:
			TPRate.append(0)
			FPRate.append(0)
			Precision.append(0)
			Recall.append(0)
			F1Score.append(0)
			MCC.append(0)

	WeightedAverage = []
	WeightedAverage.append(weightedAverage(TPRate, sumRows))
	WeightedAverage.append(weightedAverage(FPRate, sumRows))
	WeightedAverage.append(weightedAverage(Precision, sumRows))
	WeightedAverage.append(weightedAverage(Recall, sumRows))
	WeightedAverage.append(weightedAverage(F1Score, sumRows))
	WeightedAverage.append(weightedAverage(MCC, sumRows))

	output.write('=== Detailed Accuracy By Class ===\n\n')
	output.write('                TP Rate  FP Rate  Precision  Recall  F-Measure  MCC    Class\n')
	for i in range(size):
		output.write('                ')
		output.write('%0.3f    %0.3f    %0.3f      %0.3f   %0.3f      %0.3f  %s\n' %(TPRate[i], FPRate[i], Precision[i], Recall[i], F1Score[i], MCC[i], Target_Attribute[i]))
	output.write('Weigted Avg.    ')
	output.write('%0.3f    %0.3f    %0.3f      %0.3f   %0.3f      %0.3f\n' %(WeightedAverage[0], WeightedAverage[1], WeightedAverage[2], WeightedAverage[3], WeightedAverage[4], WeightedAverage[5]))
