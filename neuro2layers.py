#!/usr/bin/python
import numpy as np
import pdb
import math

class twolayernetwork:
	def __init__(self, inputlayerlength, hiddenlayerlength, outputlayerlength, gamma):
		self.learningset=dict()
		self.inputlayer=np.zeros( (1, inputlayerlength+1), float)
		self.hiddenlayer=np.zeros( (1, hiddenlayerlength+1), float)
		self.hiddenderiv=np.zeros( (1, hiddenlayerlength+1), float)
		self.outputlayer=np.zeros( (1, outputlayerlength), float)
		self.outputderiv=np.zeros( (1, outputlayerlength+1), float)
		self.W1=np.random.random_sample( (inputlayerlength+1, hiddenlayerlength) ) # +1 due to constant bias
		self.W2=np.random.random_sample( (hiddenlayerlength+1, outputlayerlength) ) # +1 due to constant bias
		#~ self.W1[-1,:]=0.3
		#~ self.W2[-1,:]=0.3
		self.gamma =gamma
		
		#set biases
		self.inputlayer[0,-1] = 1
		self.hiddenlayer[0,-1] = 1
		
	def __str__(self):
		#~ pdb.set_trace()
		outstr=""
		fstr="{:<"+str(self.inputlayer.shape[0])+"}::{:<"+str(self.W1.shape[1])+"}::{:<"+str(self.hiddenlayer.shape[0])+"}::{:<"+str(self.W2.shape[1])+"}::{:<"+str(self.outputlayer.shape[0])+"}"
		maxlen = max(self.inputlayer.shape[1], self.hiddenlayer.shape[1], self.outputlayer.shape[1])		
		for i in xrange(maxlen):
			outstr += fstr.format(self.inputlayer[:,i] if i<self.inputlayer.shape[1] else "", self.W1[i] if i < self.W1.shape[0] else "", self.hiddenlayer[:,i] if i < self.hiddenlayer.shape[1] else "", self.W2[i] if i < self.W2.shape[0] else "", self.outputlayer[:,i] if i < self.outputlayer.shape[1] else "")+"\n"
		
		return outstr
		
	def fermithresh(self, x):
		return 1./(np.exp(-x)+1)

	def forwardprop(self, inputarr):
		#~ pdb.set_trace()
		if type(inputarr)==np.ndarray:
			if len(inputarr) != len(self.inputlayer[0])-1:
				raise ValueError
		elif type(inputarr)==int:
			if len(self.inputlayer[0])!=2:
				raise ValueError
		else:
			raise ValueError
			
		self.inputlayer[0,:-1]=inputarr		
		self.hiddenlayer[0,:-1]=np.dot(self.inputlayer, self.W1)
		#~ pdb.set_trace()
		self.hiddenlayer[0,:-1] = self.fermithresh(self.hiddenlayer[0,:-1])
		self.hiddenderiv[0,:-1] = self.hiddenlayer[0,:-1]*(1-self.hiddenlayer[0,:-1])
		self.hiddenderiv[0, -1]=1
		#~ pdb.set_trace()	
		self.outputlayer=np.dot(self.hiddenlayer, self.W2)
		self.outputlayer[0,:-1] = self.fermithresh(self.outputlayer[0,:-1])
		self.outputderiv[0]=self.outputlayer[0]*(1-self.outputlayer[0])
		#~ pdb.set_trace()
	def setlearningset(self, learnset):
		self.learningset.update(learnset)
			
	def calcerror(self):
		#~ pdb.set_trace()
		totalerror=0
		for key in self.learningset.keys():
			self.forwardprop(key)
			totalerror+=((self.outputlayer-self.learningset[key])**2).sum()
		totalerror/=len(self.learningset)	
		return totalerror
				
	def backprop(self):		
		#~ pdb.set_trace()
		dW1=np.zeros((self.W1.shape[0], self.W1.shape[1]), float)
		dW2=np.zeros((self.W2.shape[0], self.W2.shape[1]), float)
		for key in self.learningset.keys():
			#~ pdb.set_trace()
			self.forwardprop(key)
			D2=np.identity(self.outputlayer.shape[1])*self.outputlayer
			D1=np.identity(self.hiddenlayer.shape[1]-1)*self.hiddenlayer[0,:-1]
			#~ pdb.set_trace()
			e=(self.outputlayer-self.learningset[key])**2
			delta2=np.dot(D2, e)
			delta1=np.dot(np.dot(D1,self.W2[:-1]), delta2)
			#~ delta2=self.outputderiv[:-1]*(self.outputlayer-self.learningset[key]).T
			#~ delta1=self.hiddenderiv[:]*np.dot(self.W2, delta2)
			dW2+=(delta2*self.hiddenlayer).T
			dW1+=(delta1*self.inputlayer).T

		self.W1-=self.gamma*dW1
		self.W2-=self.gamma*dW2
