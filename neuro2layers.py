#!/usr/bin/python
import numpy as np
import pdb
import math

class twolayernetwork:
	def __init__(self, inputlayerlength, hiddenlayerlength, outputlayerlength, gamma):
		self.learningset=dict()
		self.inputlayer=np.zeros( (1, inputlayerlength+1), float)
		self.hiddenlayer=np.zeros( (1, hiddenlayerlength+1), float)
		self.hiddenderiv=np.zeros( (hiddenlayerlength, 1), float)
		self.outputlayer=np.zeros( (1, outputlayerlength), float)
		self.outputderiv=np.zeros( (outputlayerlength, 1), float)
		self.W1=np.random.random_sample( (inputlayerlength+1, hiddenlayerlength) ) # +1 due to constant bias
		self.W2=np.random.random_sample( (hiddenlayerlength+1, outputlayerlength) ) # +1 due to constant bias
		self.W1[-1,:]=0.3
		self.W2[-1,:]=0.3
		self.gamma =gamma

	def fermithresh(self, x):
		return 1./(math.exp(-x)+1)

	def forwardprop(self, inputarr):
		pdb.set_trace()
		if len(inputarr) != len(self.inputlayer[0])-1:
			raise ValueError
		self.inputlayer[0,:-1]=inputarr		
		self.hiddenlayer[0,:-1]=np.dot(self.inputlayer, self.W1)
		#~ pdb.set_trace()
		for i in xrange(self.hiddenlayer.shape[1]):
			self.hiddenlayer[0,i]=self.fermithresh(self.hiddenlayer[0,i])
		self.hiddenderiv[:,0]=self.hiddenlayer[0,:-1]*(1-self.hiddenlayer[0,:-1])
		#~ pdb.set_trace()	
		self.outputlayer=np.dot(self.hiddenlayer, self.W2)
		for i in xrange(self.outputlayer.shape[1]):
			self.outputlayer[0,i]=self.fermithresh(self.outputlayer[i])
		self.outputderiv[:,0]=self.outputlayer[0,:]*(1-self.outputlayer[0,:])
		#~ pdb.set_trace()
	def setlearningset(self, learnset):
		self.learningset.update(learnset)
			
	def calcerror(self):
		totalerror=0
		for key in self.learningset.keys():
			self.forwardprop(key)
			totalerror+=((self.outputlayer[0]-self.learningset[key])**2).sum()
		totalerror/=len(self.learningset)	
		return totalerror
				
	def backprop(self):
		#~ pdb.set_trace()
		dW1=np.zeros((self.W1.shape[0]-1, self.W1.shape[1]), float)
		dW2=np.zeros((self.W2.shape[0]-1, self.W2.shape[1]), float)
		for key in self.learningset.keys():
			self.forwardprop(key)
			delta2=self.outputderiv*(self.outputlayer-self.learningset[key]).T
			delta1=self.hiddenderiv*np.dot(self.W2[:-1], delta2)
			dW2+=(delta2*self.hiddenlayer[0][:-1]).T
			dW1+=(delta1*self.inputlayer[0][:-1]).T

		self.W1[:-1,:]-=self.gamma*dW1
		self.W2[:-1,:]-=self.gamma*dW2
