#!/usr/bin/python
# encoding:utf-8
import numpy as np
from scipy.optimize import minimize
from scipy.optimize import basinhopping
import matplotlib.pylab as plt
import cPickle as pickle
import ipdb


def sigmoid(x):
    return 1./(1+np.exp(-x))

    
def identity(x):
    return x


def basinhopping_wrapper(*args, **kwargs):
    if "args" in kwargs:
        args = kwargs.pop("args")
    if "options" in kwargs:
        options = kwargs.pop("options")
        if "disp" in options:
            disp = options["disp"]
        else:
            disp = False
    if "fun" in kwargs:
        func = kwargs.pop("fun")
    if "method" in kwargs:
        kwargs.pop("method")
    return basinhopping(func=func, minimizer_kwargs=dict(args=args), disp=disp, **kwargs)


class FeedForwardNetwork:
    def __init__(self, layer_lengths, bias=True, hidden_activation=np.tanh, output_activation=identity, 
                 weight_range=(0.0, 1.0), verbose=False, chunk_size=None, opt_method="BFGS", filename=None):
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
        self.weight_range = weight_range
        if not filename:
            self.filename = "neural_net_" + "_".join(map(str, [ll for ll in self.layer_lengths]))
        else:
            self.filename = filename


        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        
        self.activation_fcts = [hidden_activation for layer in self.layer_lengths]
        self.activation_fcts.append(self.output_activation)
        
        # Initialize one-dimensional weight array. This array contains all weights of the neural network
        # The array can be passed to the Scipy optimization routines.
        # For matrix multiplication between layers, views on the array will be provided
        weight_array_length = 0
        
        for hl_prev, hl_next in zip(self.layer_lengths[:-1], self.layer_lengths[1:]):
            weight_array_length += (hl_prev+bias) * hl_next
            
        self.weight_array = np.random.uniform(weight_range[0], weight_range[1], size=weight_array_length)

        self.weights = []
        start = 0
        for hl_prev, hl_next in zip(self.layer_lengths[:-1], self.layer_lengths[1:]):
            # if bias is activated, bias weights will be added to the weight matrices
            end = start + (hl_prev+bias) * hl_next
            hl_weight = self.weight_array[start:end].reshape((hl_prev+bias, hl_next))
            self.weights.append(hl_weight)
            start = end
            
        # the bias weights are left out from the matrix multiplication
        if self.bias:
            self.weight_slicer = slice(None, -1)
        else:
            self.weight_slicer = slice(None, None)
        
        if self.chunk_size:
            self._prop = self._forward_prop_chunked
        else:
            self._prop = self._forward_prop
            
    @classmethod
    def from_file(cls, filename, *args, **kwargs):
        """Expects a filename from which to load the weights, and all args and kwargs 
        that would be used in the __init__ method"""
        with open(filename, "rb") as f:
            data = pickle.load(f)
        bias = data["bias"]
        kwargs["bias"] = bias
        layer_lengths = data["layer_lengths"]
        neuro = cls(layer_lengths, **kwargs)
        neuro.weight_array[:] = data["weight_array"]
        return neuro

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
            input_array = np.dot(input_array, weight[self.weight_slicer])
            if self.bias:
                input_array += weight[-1]
            input_array = act_fct(input_array)
        return input_array

    def _forward_prop_chunked(self, input_array, weights):
        for weight, act_fct in zip(weights, self.activation_fcts):
            temp_arr = np.zeros((input_array.shape[0], weight[self.weight_slicer].shape[1]), 
                                dtype=input_array.dtype)
            
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

    def calc_error(self, weight_array, inputs, outputs):
        """Returns the mean squared deviation of the neural network's output from a training set"""
        self.weight_array[:] = weight_array
        nn_output = self._prop(inputs, self.weights)
        error = ((nn_output - outputs)**2).sum()
        return error

    def optimize(self, (inputs, outputs), attempts=100, basin_steps=100):
        """Expects a tuple consisting of an array of input values and an array of output values.
        The weights are the optimized until the squared deviation of the neural network's output from the output
        values becomes minimal."""
        result_collection = []
        global minimize
        
        for attempt in xrange(attempts):
            self.weight_array[:] = np.random.uniform(self.weight_range[0], self.weight_range[1], 
                                                     self.weight_array.size)
            if self.optimization_method == "basin":
                results = basinhopping(func=self.calc_error, x0=self.weight_array, 
                                       minimizer_kwargs=dict(args=(inputs, outputs)), 
                                       disp=True, niter=basin_steps
                                       )
                xval = results.x
                fval = results.fun
            else:
                results = minimize(fun=self.calc_error, x0=self.weight_array, args=(inputs, outputs), 
                                   method=self.optimization_method, options={"disp":True})
                xval = results["x"]
                fval = results["fun"]
            with open("{}_temp".format(self.filename), "wb") as f:
                if self.verbose:
                    print "writing temporary results to {}_temp".format(self.filename)
                pickle.dump(result_collection, f)
        self.weight_array[:] = min(result_collection, key=lambda x:x[1])[0]
        
    def save_weights(self):
        data = dict()
        data["layer_lengths"] = self.layer_lengths
        data["bias"] = self.bias
        data["weight_array"] = self.weight_array
        with open(self.filename, "wb") as f:
            pickle.dump(data, f)
    


def test():
    def f(x):
        return np.sin(x)*np.exp(-x**2/100)

    inputs = np.linspace(-10, 10, 25)[:, None]
    outputs = f(inputs)
    print inputs.shape
    print outputs.shape

    answer = raw_input("Load from file?\n")
    if answer.lower() in ["y", "yes"]:
        fname = raw_input("Filename?\n")
        nn = FeedForwardNetwork.from_file(fname)
    else:
        nn = FeedForwardNetwork([1, 5, 1], verbose=False)    
        nn.optimize((inputs, outputs), attempts=10)
    
    x = np.linspace(-15, 15, 100)
    y = nn.sim(x[:, None])

    plt.plot(x, f(x), "x")
    plt.plot(inputs, f(inputs), 'ro', label="training data")
    plt.plot(x, y, label="NN output")
    plt.legend(loc="upper left")
    plt.show()
    
    answer = raw_input("Save Neural Network?\n")
    if answer.lower() in ["y", "yes"]:
        nn.save_weights()

if __name__ == "__main__":
    test()
