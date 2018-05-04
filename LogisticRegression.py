import math
import json
import numpy as np
import mapFeature

def loadConfig(jsonFilename):
	config = json.loads(open(jsonFilename).read())
	dataFilename = config['Dataset']
	data = open(dataFilename).readlines()
	
	dataset = np.loadtxt(dataFilename, delimiter=",")
	return dataset, config['Theta'], config['Alpha'], config['Lambda'], config['NumIter']

def sigmoid(X, theta):
	return 1.0/(1.0 + np.e**(-(X.dot(theta)))) 

class logisticRegression:
	def __init__(self, dataset, theta, alpha, Lambda, numIter):
		self.X = mapFeature.mapFeature(dataset[:,0], dataset[:,1])

		self.m = self.X.shape[0]
		self.n = self.X.shape[1]
		self.theta = np.array(theta).reshape([self.n,1])
		self.alpha = alpha
		self.Lambda = 0.0005
		self.numIter = 1000000

		self.Y = dataset[:,-1].reshape([self.m,1])
		self.gradientDescent()

	def computeCost(self):
		H = sigmoid(self.X, self.theta)
		H1 = np.log(H)
		H2 = np.log(1.0 - H)
		return (-self.Y.T.dot(H1) - (1.0 - self.Y).T.dot(H2))[0,0]/(1.0*self.m) + (self.theta.T.dot(self.theta)[0,0])*self.Lambda/(2.0*self.m)

	def gradientDescent(self):
		for i in range(self.numIter):
			H = sigmoid(self.X, self.theta)
			A = H - self.Y
			grad = self.X.T.dot(A)/(1.0*self.m)
			regular = self.theta
			regular[0,0] = 0
			grad = grad + self.Lambda*regular/(1.0*self.m)

			self.theta = self.theta - self.alpha*grad
			if i % 100000 == 0:
				print('i = %4d   Cost = %0.3f' % (i, self.computeCost()))

	def predict(self, jsonAccuracy):
		H = sigmoid(self.X, self.theta)
		Y_predict = np.zeros(self.Y.shape)
		for i in range(self.m):
			if H[i, 0] >= 0.5:
				Y_predict[i] = 1
		#print(Y_predict)

		nTrue = 0
		for i in range(self.m):
			if Y_predict[i,0] == self.Y[i,0]:
				nTrue += 1
		accuracy = 1.0*nTrue / (1.0*self.m)

		out = {'Accuracy':accuracy}
		with open(jsonAccuracy, 'w') as fp:
			json.dump(out, fp)
		return Y_predict, accuracy

	def save(self, outFilename):
		data = {}
		data['Theta'] = self.theta.reshape(self.n).tolist()
		data['Cost'] = self.computeCost()
		with open(outFilename, 'w') as fp:
			json.dump(data, fp)

def main():
	dataset, theta, alpha, Lambda, numIter = loadConfig('config.json')
	model = logisticRegression(dataset, theta, alpha, Lambda, numIter)
	model.save('model.json')
	model.predict('accuracy.json')


if __name__ == '__main__':
	main()