#!/usr/bin/python
import math,random

class neurolayer:
	def __init__(self, inputlayer, hiddenlayer_list, outputlayer):
		self.inputlayer=[]
		self.hiddenlayers=[]
		self.outputlayer=[]
		self.errorlayer=[]
		self.trainingdata=[]
		for i in range(inputlayer):
			self.inputlayer.append(neuron())
		for i in range(len(hiddenlayer_list)):
			self.hiddenlayers.append([])
		for i in range(hiddenlayer_list[0]):
			self.hiddenlayers[0].append(neuron([1]*inputlayer, self.inputlayer))
		for i in range(1,len(hiddenlayer_list)):
			for j in range(neuronnumber[i]):
				self.hiddenlayers[i].append(neuron([1]*hiddenlayer_list[i-1], self.hiddenlayers[i-1]))
		for i in range(outputlayer):
			self.outputlayer.append(neuron([1]*hiddenlayer, self.hiddenlayer))
		for i in range(outputlayer):
			self.errorlayer.append(neuron([1], self.outputlayer[i]))
			
	def setinput(self, inputlist):
		self.inputlist=inputlist
		for i in range(len(inputlist)):
			self.inputlayer[i].value=inputlist[i]
			
	def calcoutput(self):
		for layer in self.hiddenlayers:
			for n in layer:
				n.output()
		for n in self.outputlayer:
			n.output()
	#trainingsdaten in form von dict {input:output}		
	def addtrainingdata(self, data):
		self.trainingdata.append(data)
		
	def backpropagation(self):
		
		
		
		

class neuron:
	def __init__(self, weights=None, ancestors=None, value=0):
		self.weights=weights
		if value==0:
			self.value=random.random()
		else:
			self.value=value
		if ancestors:
			self.ancestors=ancestors
	
	def getinputs(self, ancestors):
		self.ancestors=ancestors
		
	def fermithresh(self, x):
		return 1./(math.exp(-x)+1)
	
	def fermideriv(self, x):
		ex=math.exp(-x)
		return ex/((ex+1)(ex+1))
		
	def output(self):
		summe=0
		if len(self.weights)!=len(self.ancestors):
			raise Exception
		for i in range(len(self.weights)):
			summe+=self.ancestors[i].value*self.weights[i]
		self.value=self.fermithresh(summe)
		self.deriv=fermideriv(summe)

class errornode:
	def __init__(self, ):
		
