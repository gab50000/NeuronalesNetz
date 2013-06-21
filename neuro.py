#!/usr/bin/python
import math

class neuro2layer:
	def __init__(self, inputlayer, hiddenlayer, outputlayer):
		self.inputlayer=[]
		self.hiddenlayer=[]
		self.outputlayer=[]
		self.trainingdata=[]
		for i in range(inputlayer):
			self.inputlayer.append(neuron())
		for i in range(hiddenlayer):
			self.hiddenlayer.append(neuron([1]*inputlayer, self.inputlayer, 1))
		for i in range(outputlayer):
			self.outputlayer.append(neuron([1]*hiddenlayer, self.hiddenlayer, 1))
			
	def setinput(self, inputlist):
		for i in range(len(inputlist)):
			self.inputlayer[i].value=inputlist[i]
			
	def calcoutput(self):
		for neu in self.hiddenlayer:
			neu.output()
		for neu in self.outputlayer:
			neu.output()
			
	def addtrainingdata(self, data):
		self.trainingdata.append(data)
		

class neuron:
	def __init__(self, weights=None, ancestors=None, value=0):
		self.weights=weights
		self.value=value
		if ancestors:
			self.ancestors=ancestors
	
	def getinputs(self, ancestors):
		self.ancestors=ancestors
		
	def fermithresh(self, x):
		return 1./(math.exp(-x)+1)
		
	def output(self):
		summe=0
		if len(self.weights)!=len(self.ancestors):
			raise Exception
		for i in range(len(self.weights)):
			summe+=self.ancestors[i].value*self.weights[i]
		self.value=self.fermithresh(summe)
