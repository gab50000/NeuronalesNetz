#!/usr/bin/python
import numpy as np
from scipy.optimize import minimize as minimize
import matplotlib.pylab as plt
import ipdb

def sigmoid(x):
    return 1/(1+np.exp(-x))

class FeedForwardNetwork:
    def __init__(self, layer_lengths, bias=True, hidden_activation=np.tanh, output_activation=lambda x:x):
        self.layer_lengths = layer_lengths

        self.bias = bias
        # if bias is activated, the bias weights are left out from the matrix multiplication
        if self.bias:
            self.weight_slicer = slice(None, -1)
        else:
            self.weight_slicer = slice(None, None)

        self.hidden_activation = hidden_activation
        self.output_activation = output_activation

        self.weights = []
        for hl_prev, hl_next in zip(self.layer_lengths[:-1], self.layer_lengths[1:]):
            hl_weight = np.random.random((hl_prev+bias, hl_next))
            self.weights.append(hl_weight)

    def _forward_prop(self, inputarr, weights):
        for weight in weights[:-1]:
            inputarr = np.dot(inputarr, weight[self.weight_slicer])
            if self.bias:
                inputarr += weight[-1]
            inputarr = self.hidden_activation(inputarr)
        output = np.dot(inputarr, weights[-1][self.weight_slicer])
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
        start_pos = 0
        for hl_prev, hl_next in zip(self.layer_lengths[:-1], self.layer_lengths[1:]):
            shape = hl_prev+self.bias, hl_next
            weights.append(weight_array[start_pos:start_pos+shape[0]*shape[1]].reshape(shape))
            start_pos += shape[0]*shape[1]
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
    nn = FeedForwardNetwork([1, 10, 1])
    inputs = np.linspace(-10, 10, 25)[:, None]
    outputs = f(inputs)
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