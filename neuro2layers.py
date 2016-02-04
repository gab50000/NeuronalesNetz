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
        intermediate = np.dot(inputarr[:, :self.inputlayer_length], weights[0][:self.inputlayer_length])
        if self.bias:
            intermediate += weights[0][-1]
        intermediate = self.hidden_activation(intermediate)
        for weight in weights[1:-1]:
            intermediate = np.dot(intermediate[:, :self.inputlayer_length], weight[:self.inputlayer_length])
            if self.bias:
                intermediate += weight[-1]
        output = np.dot(intermediate[:, :self.inputlayer_length], weights[-1][:self.inputlayer_length])
        output = self.output_activation(output)

        return output

    def sim(self, input):
        return self._forward_prop(input, self.weights)

    def calc_error(self, weights_array, inputs, outputs):
        print weights_array
        weights = self._unflatten(weights_array)
        nn_output = self._forward_prop(inputs, weights)
        error = ((nn_output - outputs[:, 0])**2).sum()
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
        weights.append(weight_array[start_pos:start_pos+shape[0]*shape[1]])
        return weights

        # len_W1 = (self.inputlayerlength+1) * self.hiddenlayerlength
        # # len_W2 = (self.hiddenlayerlength+1) * self.outputlayerlength
        #
        # W1 = weights[:len_W1].reshape((self.inputlayerlength+1, self.hiddenlayerlength))
        # W2 = weights[len_W1:].reshape((self.hiddenlayerlength+1, self.outputlayerlength))
        #
        # print "W1:"
        # print W1
        # print "W2:"
        # print W2
        #
        # totalerror = 0
        # for input, output in zip(inputs, outputs):
        #     outputlayer = self.forwardprop(input, W1, W2)
        #     totalerror += ((outputlayer - output) ** 2).sum()
        # totalerror /= inputs.shape[0]
        # return totalerror

    def optimize(self, training_set):
        inputs, outputs = training_set
        weights = np.hstack([w.flatten() for w in self.weights])
        results = minimize(fun=self.calc_error, x0=weights, args=(inputs, outputs), method='BFGS')
        result = results["x"]
        print "diff:", (weights - result).sum()
        self.weights = self._unflatten(result)
        return self.weights

# def optimize(obj, inputs, outputs):
#     weights = np.hstack([obj.W1.flatten(), obj.W2.flatten()])
#     results = minimize(fun=obj.calc_error, x0=weights, args=(inputs, outputs), method='BFGS', options={'xtol': 1e-8, 'disp': True})
#     result = results["x"]
#     obj.W1 = result[:obj.W1.size].reshape(obj.W1.shape)
#     obj.W2 = result[obj.W1.size:].reshape(obj.W2.shape)
#     x = np.linspace(-5, 5)
#     y = np.zeros(x.shape)
#     for i, v in enumerate(x):
#         y[i] = obj.forward_prop(v, obj.W1, obj.W2)
#     plt.plot(x, y)
#     plt.plot(inputs, outputs)
#     plt.show()
#     ipdb.set_trace()
#     return result["x"]

def test():
    nn = FeedForwardNetwork([1, 20, 1])
    inputs = np.linspace(-10, 10, 25)[:, None]
    outputs = inputs**2
    print inputs.shape
    print outputs.shape
    nn.optimize((inputs, outputs))

    x = np.linspace(-15, 15, 100)
    y = nn.sim(x[:, None])

    plt.plot(x, x**2, "x")
    plt.plot(x, y)
    plt.show()

    ipdb.set_trace()

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