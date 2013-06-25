#!/usr/bin/python

import numpy as np
import random

class layer:
	def __init__(shape, nodenumber):
		self.nodes=np.zeros(shape, np.dtype=float)
		self.derivs=np.zeros(shape, np.dtype=float)
		
	def fermithresh(self, x):
		return 1./(math.exp(-x)+1)
	
	def fermideriv(self, x):
		ex=math.exp(-x)
		return ex/((ex+1)(ex+1))
	
class neurolayer:
	def __init__(self, layerlengths):
		matrices=[]
		for i in range(1,len(layerlengths)):
			matrices.append(np.array([[random.random() for i in range(layerlengths[i-1])] for j in range(layerlengths[i])]))
		

