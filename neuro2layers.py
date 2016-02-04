#!/usr/bin/python
import numpy as np
from scipy.optimize import minimize as minimize
import matplotlib.pylab as plt
import ipdb


class TwoLayerNetwork:
    def __init__(self, inputlayerlength, hiddenlayerlength, outputlayerlength, gamma):
        self.inputlayerlength = inputlayerlength
        self.hiddenlayerlength = hiddenlayerlength
        self.outputlayerlength = outputlayerlength

        self.inputlayer = np.zeros((1, inputlayerlength + 1), float)
        self.hiddenlayer = np.zeros((1, hiddenlayerlength + 1), float)
        self.hiddenderiv = np.zeros((1, hiddenlayerlength + 1), float)
        self.outputlayer = np.zeros((1, outputlayerlength), float)
        self.outputderiv = np.zeros((1, outputlayerlength + 1), float)
        self.W1 = np.random.uniform(-1, 1, size=(inputlayerlength + 1, hiddenlayerlength))  # +1 due to constant bias
        self.W2 = np.random.uniform(-1, 1, size=(hiddenlayerlength + 1, outputlayerlength))  # +1 due to constant bias
        # ~ self.W1[-1,:]=0.3
        # ~ self.W2[-1,:]=0.3
        self.gamma = gamma

        # set biases
        self.inputlayer[0, -1] = 1
        self.hiddenlayer[0, -1] = 1

    def __str__(self):
        # ~ pdb.set_trace()
        outstr = ""
        fstr = "{:<" + str(self.inputlayer.shape[0]) + "}::{:<" + str(self.W1.shape[1]) + "}::{:<" + str(
            self.hiddenlayer.shape[0]) + "}::{:<" + str(self.W2.shape[1]) + "}::{:<" + str(
            self.outputlayer.shape[0]) + "}"
        maxlen = max(self.inputlayer.shape[1], self.hiddenlayer.shape[1], self.outputlayer.shape[1])
        for i in xrange(maxlen):
            outstr += fstr.format(self.inputlayer[:, i] if i < self.inputlayer.shape[1] else "",
                                  self.W1[i] if i < self.W1.shape[0] else "",
                                  self.hiddenlayer[:, i] if i < self.hiddenlayer.shape[1] else "",
                                  self.W2[i] if i < self.W2.shape[0] else "",
                                  self.outputlayer[:, i] if i < self.outputlayer.shape[1] else "") + "\n"

        return outstr

    def fermi_function(self, x):
        return 1. / (np.exp(-x) + 1)

    def forwardprop(self, inputarr, weights1, weights2):
        self.inputlayer[0, :-1] = inputarr
        self.hiddenlayer[0, :-1] = np.dot(self.inputlayer, weights1)
        self.hiddenlayer[0, :-1] = np.tanh(self.hiddenlayer[0, :-1])
        # self.hiddenderiv[0, :-1] = self.hiddenlayer[0, :-1] * (1 - self.hiddenlayer[0, :-1])
        # self.hiddenderiv[0, -1] = 1

        outputlayer = np.dot(self.hiddenlayer, weights2)
        # outputlayer = np.tanh(self.outputlayer)
        # outputlayer = self.fermi_function(self.outputlayer)
        # self.outputderiv[0] = self.outputlayer[0] * (1 - self.outputlayer[0])
        return outputlayer

    def calc_error(self, weights, inputs, outputs):
        len_W1 = (self.inputlayerlength+1) * self.hiddenlayerlength
        # len_W2 = (self.hiddenlayerlength+1) * self.outputlayerlength

        W1 = weights[:len_W1].reshape((self.inputlayerlength+1, self.hiddenlayerlength))
        W2 = weights[len_W1:].reshape((self.hiddenlayerlength+1, self.outputlayerlength))

        print "W1:"
        print W1
        print "W2:"
        print W2

        totalerror = 0
        for input, output in zip(inputs, outputs):
            outputlayer = self.forwardprop(input, W1, W2)
            totalerror += ((outputlayer - output) ** 2).sum()
        totalerror /= inputs.shape[0]
        return totalerror

    def backprop(self):
        # ~ pdb.set_trace()
        dW1 = np.zeros((self.W1.shape[0], self.W1.shape[1]), float)
        dW2 = np.zeros((self.W2.shape[0], self.W2.shape[1]), float)
        for key in self.learningset.keys():
            # ~ pdb.set_trace()
            self.forwardprop(key)
            D2 = np.identity(self.outputlayer.shape[1]) * self.outputlayer
            D1 = np.identity(self.hiddenlayer.shape[1] - 1) * self.hiddenlayer[0, :-1]
            # ~ pdb.set_trace()
            e = (self.outputlayer - self.learningset[key]) ** 2
            delta2 = np.dot(D2, e)
            delta1 = np.dot(np.dot(D1, self.W2[:-1]), delta2)
            # ~ delta2=self.outputderiv[:-1]*(self.outputlayer-self.learningset[key]).T
            # ~ delta1=self.hiddenderiv[:]*np.dot(self.W2, delta2)
            dW2 += (delta2 * self.hiddenlayer).T
            dW1 += (delta1 * self.inputlayer).T

        self.W1 -= self.gamma * dW1
        self.W2 -= self.gamma * dW2

def optimize(obj, inputs, outputs):
    weights = np.hstack([obj.W1.flatten(), obj.W2.flatten()])
    results = minimize(fun=obj.calc_error, x0=weights, args=(inputs, outputs), method='BFGS', options={'xtol': 1e-8, 'disp': True})
    result = results["x"]
    obj.W1 = result[:obj.W1.size].reshape(obj.W1.shape)
    obj.W2 = result[obj.W1.size:].reshape(obj.W2.shape)
    x = np.linspace(-5, 5)
    y = np.zeros(x.shape)
    for i, v in enumerate(x):
        y[i] = obj.forwardprop(v, obj.W1, obj.W2)
    plt.plot(x, y)
    plt.plot(inputs, outputs)
    plt.show()
    ipdb.set_trace()
    return result["x"]

def test():
    nn = TwoLayerNetwork(inputlayerlength=1, hiddenlayerlength=10, outputlayerlength=1, gamma=1)
    inputs = np.linspace(-10, 10)
    outputs = inputs**2
    # outputs = np.sin(inputs)  # + np.random.uniform(-.1, -1, size=inputs.shape)
    optimize(nn, inputs, outputs)

# def optitest():
#     def devsquare(vars, x, y):
#         m, y0 = vars
#         deviation = ((m*x+y0 - y)**2).sum()
#         return deviation
#
#     x = np.linspace(0, 10)
#     y = 3*x + 2.5 + np.random.uniform(-1, 1, size=x.shape)
#
#     results = minimize(devsquare, x0=(0, 0), args=(x, y))
#     ipdb.set_trace()


if __name__ == "__main__":
    test()