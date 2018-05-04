import math
import json
import numpy as np

def loadConfig(jsonFilename):
	config = json.loads(open(jsonFilename).read())
	dataFilename = config['Dataset']
	data = open(dataFilename).readlines()
	
	dataset = np.loadtxt(dataFilename, delimiter=",")
	return dataset, config['Theta'], config['Alpha'], config['NumIter']

class linearRegression:
	def __init__(self, dataset, theta, alpha, numIter):
		self.dataset = dataset.astype(np.float)
		self.m = dataset.shape[0]
		self.n = dataset.shape[1]
		self.theta = np.array(theta).reshape([self.n,1])
		self.alpha = 0.01
		self.numIter = numIter

		self.mu = np.zeros([self.n, 1])
		self.std = np.zeros([self.n, 1])
				
		for i in range(self.n):
			self.mu[i] = np.mean(dataset[:,i])
			self.std[i] = np.std(dataset[:,i])

		self.normalizeFeature()
		self.X = self.dataset[:,:-1]
		self.Y = self.dataset[:,-1].reshape(self.m, 1)
		self.X = np.c_[np.ones([self.m, 1]), self.X]

		self.gradientDescent()


	def normalizeFeature(self):
		for i in range(self.n):
			self.dataset[:,i] = (self.dataset[:,i] - self.mu[i,0])/self.std[i,0]

	def computeCost(self):
		A = self.X.dot(self.theta) - self.Y
		return (A.T.dot(A))[0,0]/(self.m*2)

	def gradientDescent(self):
		grad = np.zeros(self.theta.shape)
		for i in range(self.numIter):
			A = self.X.dot(self.theta) - self.Y
			grad = self.X.T.dot(A)
			self.theta = self.theta - self.alpha*grad
			if i % 100 == 0:
				print('i = %4d   Cost = %0.3f' % (i, self.computeCost()))

	def predict(self, X):
		X = X.reshape([1, self.n - 1]).astype(np.float)
		for i in range(X.shape[1]):
			X[0,i] = ((X[0,i] - self.mu[i,0])/self.std[i,0])
		X = np.c_[np.ones([1,1]), X]
		return (X.dot(self.theta))[0,0]*self.std[-1,0] + self.mu[-1, 0]

	def save(self, outFilename):
		data = {}
		data['Theta'] = self.theta.reshape(self.n).tolist()
		data['Cost'] = self.computeCost()
		with open(outFilename, 'w') as fp:
			json.dump(data, fp)

def predict(model, jsonInput):
	input = json.loads(open('price.json').read())
	X = np.array([input['Size'], input['Bedroom']])

	input['Price'] = np.round(model.predict(X).round())
	with open(jsonInput, 'w') as fp:
		json.dump(input, fp)
	return input['Price'].astype(np.float)


def main():
	dataset, theta, alpha, numIter = loadConfig('config.json')
	model = linearRegression(dataset, theta, alpha, numIter)
	model.save('model.json')
	predict(model, 'price.json')


if __name__ == '__main__':
	main()