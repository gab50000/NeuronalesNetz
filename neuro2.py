#!/usr/bin/python

import numpy as np
import random

class layer:
	def __init__(shape, nodenumber):
		self.nodes=np.zeros((1,shape), np.dtype=float)
		self.derivs=np.zeros((1,shape), np.dtype=float)
		
	def fermithresh(self, x):
		return 1./(math.exp(-x)+1)
	
	def fermideriv(self, x):
		ex=math.exp(-x)
		return ex/((ex+1)(ex+1))
	
class neurolayer:
	def __init__(self, layerlengths):
		self.learningset=dict()
		self.matrices=[]
		self.layers=[]
		self.dEdw=[]
		self.layers.append(layer(layerlengths[0]))
		for i in range(1,len(layerlengths)):
			#~ self.matrices.append(np.matrix([[random.random() for i in range(layerlengths[i-1])] for j in range(layerlengths[i])]))
			self.matrices.append(np.random.random((layerlengths[i], layerlengths[i-1]))
			self.layers.append(layer(layerlengths[i]))
			self.dEdw.append(np.matrix(np.zeros((layerlengths[i], layerlengths[i-1]))))

	def forwardprop(self,eingabe):
		if len(eingabe)==len(self.layers[0].nodes):
			self.layers[0].nodes=eingabe
			for i in range(1,len(layers)):
				self.layers[i].nodes=np.dot(self.matrices[i-1],self.layers[i-1].nodes)
				for j in range(len(self.layers[i].nodes)):
					self.layers[i].nodes[j]=fermithresh(self.layers[i].nodes[j])
					self.layers[i].derivs[j]=self.layers[i].nodes[j]*(1-self.layers[i].nodes[j])		
		else:
			raise Exception
	
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
		for key in self.learningset.keys():
				self.forwardprop(key)
				#dEdw ist matrix mit dimension len(vorgaengerlayer)xlen(nachfolgelayer)
				self.dEdw[-1]=(self.layers[-1].nodes-self.learningset[key]).transpose()*(self.layers[-1].derivs*self.layers[-2].nodes)
		for i in range(1,len(self.dEdw)):
			self.dEdw[-i-1].fill(0)
			
		
		

