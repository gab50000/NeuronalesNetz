#!/usr/bin/python
# encoding:utf-8
import numpy as np
from scipy.optimize import minimize as minimize
import matplotlib.pylab as plt
import cPickle as pickle
import ipdb


def sigmoid(x):
    return 1./(1+np.exp(-x))

    
def identity(x):
    return x


class FeedForwardNetwork:
    def __init__(self, layer_lengths, bias=True, hidden_activation=np.tanh, output_activation=identity, verbose=False, chunk_size=None, opt_method="BFGS"):
        """A FeedForwardNetwork object is initialized by providing a list that contains the number of nodes for
        each layer.
        For example, a FFN object with an input layer with one node, a hidden layer with 5 nodes and an output layer
         with 2 nodes is initialized via

            ffn = FeedForwardNetwork([1, 5, 2])

        Further (optional) parameters are
          * bias (expects a boolean): each layer receives an additional bias node
          * hidden_activation: determines the activation function for each hidden layer
          * output_activation: determines the activation function for the output layer
          * chunks: calculate the inputs in chunks. Useful for large input data.
          """
        self.layer_lengths = layer_lengths
        self.bias = bias
        self.verbose = verbose
        self.optimization_method = opt_method
        self.chunk_size = chunk_size
        # if bias is activated, the bias weights are left out from the matrix multiplication
        if self.bias:
            self.weight_slicer = slice(None, -1)
        else:
            self.weight_slicer = slice(None, None)

        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        
        self.activation_fcts = [hidden_activation for layer in self.layer_lengths]
        self.activation_fcts.append(self.output_activation)

        self.weights = []
        for hl_prev, hl_next in zip(self.layer_lengths[:-1], self.layer_lengths[1:]):
            hl_weight = np.random.random((hl_prev+bias, hl_next))
            self.weights.append(hl_weight)
        
        if self.chunk_size:
            self._prop = self._forward_prop_chunked
        else:
            self._prop = self._forward_prop
            
    def _determine_free_ram(self):
        with open("/proc/sysinfo", "r") as f:
            info = f.readlines()
        free_bytes = int(info[0].split()[1]) * 1024
        return free_bytes
        
    def _determine_necessary_ram(self, input_length, dtype_bitsize=32):
        total = 0
        for length in self.layer_lengths:
            total += input_length * layer_length * dtype_size / 8.0
        return total

    def _forward_prop(self, input_array, weights):
        for weight, act_fct in zip(weights, self.activation_fcts):
            # print "hidden"
            input_array = np.dot(input_array, weight[self.weight_slicer])
            if self.bias:
                input_array += weight[-1]
            input_array = act_fct(input_array)
        return input_array

    def _forward_prop_chunked(self, input_array, weights):
        for weight, act_fct in zip(weights, self.activation_fcts):
            temp_arr = np.zeros((input_array.shape[0], weight[self.weight_slicer].shape[1]), dtype=input_array.dtype)
            
            for chunk in xrange(0, temp_arr.shape[0], self.chunk_size):
                if chunk + self.chunk_size > temp_arr.shape[0]:
                    sl = slice(chunk, None)
                else:
                    sl = slice(chunk+self.chunk_size)
                temp_arr[sl] = np.dot(input_array[sl], weight[self.weight_slicer])
            input_array = temp_arr
            if self.bias:
                input_array += weight[-1]
            input_array = act_fct(input_array)
        return input_array
        
    def sim(self, input_array):
        """Calculates the output for a given array of inputs.
        Expects an array of the shape (No. of inputs, No. of input nodes)"""
        return self._prop(input_array, self.weights)

    # def _calc_error(self, weight_array, inputs, outputs):
        # weights = self._unflatten(weight_array)
        # nn_output = self._prop(inputs, weights)
        # error = ((nn_output - outputs)**2).sum()
        # return error

    def calc_error(self, weight_vector, inputs, outputs):
        """Returns the mean squared deviation of the neural network's output from a training set"""
        self.weight_vector[:] = weight_vector
        nn_output = self._prop(inputs, self.weights)
        error = ((nn_output - outputs)**2).sum()
        return error
        
    def _unflatten(self, weight_array):
        weights = []
        start_pos = 0
        for hl_prev, hl_next in zip(self.layer_lengths[:-1], self.layer_lengths[1:]):
            shape = hl_prev+self.bias, hl_next
            weights.append(weight_array[start_pos:start_pos+shape[0]*shape[1]].reshape(shape))
            start_pos += shape[0]*shape[1]
        return weights

    def optimize(self, (inputs, outputs)):
        """Expects a tuple consisting of an array of input values and an array of output values.
        The weights are the optimized until the squared deviation of the neural network's output from the output
        values becomes minimal."""
        # weights = np.hstack([w.flatten() for w in self.weights])
        results = minimize(fun=self.calc_error, x0=self.weight_vector, args=(inputs, outputs), method=self.optimization_method, options={"disp":True})
        result = results["x"]
        self.weight_vector[:] = result
        # self.weights = self._unflatten(result)
        # return self.weights
        
    def save_weights(self, filename=None):
        if not filename:
            filename = "neural_net_" + "_".join(map(str, [ll for ll in self.layer_lengths]))
        with open(filename, "wb") as f:
            pickle.dump(self.weights, f)
            
    def load_weights(self, filename):
        with open(filename, "rb") as f:
            self.weights = pickle.load(f)
        self.layer_lengths = [w.shape[0]-self.bias for w in self.weights]


def test():
    def f(x):
        return np.sin(x)*np.exp(-x**2/100)
    nn = FeedForwardNetwork([1, 12, 1], verbose=False, chunk_size=3)
    inputs = np.linspace(-10, 10, 25)[:, None]
    outputs = f(inputs)
    print inputs.shape
    print outputs.shape
    nn.optimize((inputs, outputs))
    
    x = np.linspace(-15, 15, 100)
    y = nn.sim(x[:, None])

    plt.plot(x, f(x), "x")
    plt.plot(x, y, label="bla")
    plt.legend(loc="upper left")
    plt.show()

if __name__ == "__main__":
    test()
