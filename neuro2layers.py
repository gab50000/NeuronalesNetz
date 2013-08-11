#!/usr/bin/python
import numpy as np

class twolayernetwork:
	def __init__(self, inputlayerlength, hiddenlayerlength, outputlayerlength, gamma):
		self.learningset=dict()
		self.inputlayer=np.zeros( (1, inputlayerlength), float)
		self.hiddenlayer=np.zeros( (1, hiddenlayerlength), float)
		self.hiddenderiv=np.zeros( (hiddenlayerlength, 1), float)
		self.outputlayer=np.zeros( (1, outputlayerlength), float)
		self.outputderiv=np.zeros( (outputlayerlength, 1), float)
		self.W1=np.random.random_sample( (inputlayerlength+1, hiddenlayerlength) ) # +1 due to constant bias
		self.W2=np.random.random_sample( (hiddenlayerlength+1, outputlayerlength) ) # +1 due to constant bias
		self.gamma =gamma

	def fermithresh(self, x):
		return 1./(math.exp(-x)+1)

	def forwardprop(self):
		
		self.hiddenlayer=np.dot(self.W1, self.inputlayer)
		for i in xrange(hiddenlayer.shape[0]):
			self.hiddenlayer[i]=self.fermithresh(self.hiddenlayer[i])
			self.hiddenderiv[i]=self.hiddenlayer[i]*(1-self.hiddenlayer[i])
			
		self.outputlayer=np.dot(self.W2, self.hiddenlayer)
		for i in xrange(outputlayer.shape[0]):
			self.outputlayer[i]=self.fermithresh(self.outputlayer[i])
			self.outputderiv[i]=self.outputlayer[i]*(1-self.outputlayer[i])
			
	def setlearningset(self, learnset):
		self.learningset.update(learnset)
			
	def calcerror(self):
		totalerror=0
		for key in self.learningset.keys():
			self.forwardprop(key)
			for i in range(len(self.layers[-1].nodes)):
				totalerror+=(self.layers[-1].nodes[i]-self.learningset[key])**2
		totalerror/=len(self.learningset)	
				
	def backprop(self):
		dW1=np.zeros(self.W1.shape, float)
		dW2=np.zeros(self.W2.shape, float)
		
		for key in self.learningset.keys():
			self.forwardprop(key)
			delta2=self.outputderiv*(self.outputlayer-learningset[key]).T
			delta1=self.hiddenderiv*np.dot(self.W2, delta2)
			dW2+=(delta2*self.hiddenlayer).T
			dW1+=(delta1*self.inputlayer).T
		
		self.W1-=self.gamma*dW1
		self.W2-=self.gamma*dW2
