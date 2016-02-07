#!/usr/bin/python
import numpy as np
from scipy.optimize import minimize as minimize
import matplotlib.pylab as plt
import ipdb

def sigmoid(x):
    return 1/(1+np.exp(-x))

class FeedForwardNetwork:
    def __init__(self, layer_lengths, bias=True, hidden_activation=np.tanh, output_activation=lambda x:x):
        self.inputlayer_length = layer_lengths[0]
        self.hiddenlayer_lengths = layer_lengths[1:-1]
        self.outputlayer_length = layer_lengths[-1]
        self.layer_lengths = layer_lengths

        self.bias = bias
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation

        self.weights = []
        weight_input_hidden = np.random.random((self.inputlayer_length+bias, self.hiddenlayer_lengths[0]))
        self.weights.append(weight_input_hidden)
        for hl_prev, hl_next in zip(self.hiddenlayer_lengths[:-1], self.hiddenlayer_lengths[1:]):
            hl_weight = np.random.random((hl_prev+bias, hl_next))
            self.weights.append(hl_weight)
        weight_hidden_output = np.random.random((self.hiddenlayer_lengths[-1]+bias, self.outputlayer_length))
        self.weights.append(weight_hidden_output)

    def _forward_prop(self, inputarr, weights):
        for weight, layer_len in zip(weights, self.layer_lengths[:-2]):
            inputarr = np.dot(inputarr, weight[:layer_len])
            if self.bias:
                inputarr += weight[-1]
            inputarr = self.hidden_activation(inputarr)
        output = np.dot(inputarr, weights[-1][:self.layer_lengths[-2]])
        if self.bias:
            output += weights[-1][-1]
        output = self.output_activation(output)
        return output

    def sim(self, input):
        return self._forward_prop(input, self.weights)

    def calc_error(self, weights_array, inputs, outputs):
        weights = self._unflatten(weights_array)
        nn_output = self._forward_prop(inputs, weights)
        error = ((nn_output - outputs)**2).mean()
        return error

    def _unflatten(self, weight_array):
        weights = []
        shape = self.inputlayer_length + self.bias, self.hiddenlayer_lengths[0]
        weights.append(weight_array[:shape[0]*shape[1]].reshape(shape))
        start_pos = shape[0]*shape[1]
        for hl_prev, hl_next in zip(self.hiddenlayer_lengths[:-1], self.hiddenlayer_lengths[1:]):
            shape = hl_prev+self.bias, hl_next
            weights.append(weight_array[start_pos:start_pos+shape[0]*shape[1]].reshape(shape))
            start_pos += shape[0]*shape[1]
        shape = self.hiddenlayer_lengths[-1] + self.bias, self.outputlayer_length
        weights.append(weight_array[start_pos:start_pos+shape[0]*shape[1]].reshape(shape))
        return weights

    def optimize(self, training_set):
        inputs, outputs = training_set
        weights = np.hstack([w.flatten() for w in self.weights])
        results = minimize(fun=self.calc_error, x0=weights, args=(inputs, outputs), method='BFGS')
        result = results["x"]
        print "diff:", (weights - result).sum()
        self.weights = self._unflatten(result)
        return self.weights

def test():
    def f(x):
        return np.sin(x)*np.exp(-x**2/100)
    nn = FeedForwardNetwork([1, 5, 1])
    inputs = np.linspace(-10, 10, 25)[:, None]
    outputs = f(inputs)
    # outputs = inputs**2
    print inputs.shape
    print outputs.shape
    nn.optimize((inputs, outputs))

    x = np.linspace(-15, 15, 100)
    y = nn.sim(x[:, None])

    plt.plot(x, f(x), "x")
    plt.plot(x, y)
    plt.show()

if __name__ == "__main__":
    test()