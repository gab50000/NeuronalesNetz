#!/usr/bin/python
# encoding:utf-8
import argparse
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
    def __init__(self, layer_lengths, bias=True, hidden_activation=sigmoid, 
                 output_activation=identity, regularization_parameter=0, 
                 chunk_size=None, filename=None, verbose=False):
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
        self.regularization_parameter = float(regularization_parameter)
        self.verbose = verbose
        self.chunk_size = chunk_size
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
            
        self.weight_array = np.empty(weight_array_length)

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
       
        self.initialize_weights()
        
        if verbose:
            self.print_settings()
            
    def initialize_weights(self):
        for weight in self.weights:
            shape = weight[self.weight_slicer].shape
            weight[self.weight_slicer] = np.random.standard_normal(shape) / np.sqrt(shape[0])
            if self.bias:
                weight[-1] = np.random.standard_normal(shape[1])
            
            
    def print_settings(self):
        print "Layers:", self.layer_lengths 
        print "Bias:", self.bias 
        print "Lambda:", self.regularization_parameter 
            
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
        error = ((nn_output - outputs)**2).sum() / (2*outputs.shape[0]) \
                + self.regularization_parameter / (2*outputs.shape[0]) * (weight_array**2).sum()
        return error
        
    def cross_entropy(self, weight_array, inputs, outputs):
        """Returns the cross entropy of the neural network's output from a training set"""
        self.weight_array[:] = weight_array
        nn_output = self._prop(inputs, self.weights)
        error = -(outputs * np.log(nn_output) + (1-outputs) * np.log(1-nn_output)).sum() \
                + self.regularization_parameter / (2*outputs.shape[0]) * (weight_array**2).sum()
        return error
        
    def calc_weighted_error(self, weight_array, inputs, outputs):
        """Returns the mean squared deviation of the neural network's output from a training set.
           Weights the deviation of each data point.
           Needs output values in outputs[0] and weights in outputs[1]"""
        self.weight_array[:] = weight_array
        nn_output = self._prop(inputs, self.weights)
        error = (outputs[1] * (nn_output - outputs[0])**2).sum() / (2*outputs.shape[1]) \
                + self.regularization_parameter / (2*outputs.shape[1]) * (weight_array**2).sum()
        return error

    def optimize(self, (inputs, outputs), validation_set=None, attempts=100, 
                 basin_steps=100, optimization_method="BFGS", error_determination="squared_sum"):
        """Expects a tuple consisting of an array of input values and an array of output values.
        The weights are the optimized until the squared deviation of the neural network's output from the output
        values becomes minimal."""
        result_collection = []
        if validation_set:
            validation_input, validation_output = validation_set
            
        if error_determination == "squared_sum":
            error_func = self.calc_error
        elif error_determination == "cross_entropy":
            error_func = self.cross_entropy
        elif error_determination == "weighted_error":
            error_func = self.calc_weighted_error
        else:
            raise NotImplementedError("Unknown error function '{}'".format(error_determination))
        
        if inputs.shape[-1] != self.layer_lengths[0]:
            raise TypeError("Shape of input array ({}) does not match number of input layers ({})".format(inputs.shape[-1], self.layer_lengths[0]))
        
        for attempt in xrange(attempts):
            if self.verbose:
                print "Optimization", attempt
            self.initialize_weights()
            if optimization_method == "basin":
                results = basinhopping(func=error_func, x0=self.weight_array, 
                                       minimizer_kwargs=dict(args=(inputs, outputs)), 
                                       disp=self.verbose, niter=basin_steps
                                       )
                xval = results.x
                fval = results.fun
            else:
                results = minimize(fun=error_func, x0=self.weight_array, args=(inputs, outputs), 
                                   method=optimization_method, options={"disp":True})
                xval = results["x"]
                fval = results["fun"]
                
                if self.verbose:
                    print results

            if validation_set:
                mean_error_validation_set = error_func(self.weight_array, validation_input, validation_output) #((self.sim(validation_input) - validation_output)**2).mean()
                result_dict = dict(weights=xval, f=fval, 
                                   validation_error=mean_error_validation_set
                                   )
                print "Mean error validation set:", mean_error_validation_set
            else:
                result_dict = dict(weights=xval, f=fval)
                
            result_collection.append(result_dict)
            with open("{}_temp".format(self.filename), "wb") as f:
                if self.verbose:
                    print "writing temporary results to {}_temp".format(self.filename)
                pickle.dump(result_collection, f)
        if validation_set:
            self.weight_array[:] = min(result_collection, key=lambda x:x["validation_error"])["weights"]
        else:
            self.weight_array[:] = min(result_collection, key=lambda x:x["f"])["weights"]
        
    def save_weights(self):
        data = dict()
        data["layer_lengths"] = self.layer_lengths
        data["bias"] = self.bias
        data["weight_array"] = self.weight_array
        
        if os.path.exists(self.filename):
            i = 1
            fn, ext = os.path.splitext(self.filename)
            while os.path.exists(self.filename):
                print "File", self.filename, "already exists"
                self.filename = "{}({:02d}){}".format(fn, i, ext)
                i += 1
        
        print "Saving weights as", self.filename
        with open(self.filename, "wb") as f:
            pickle.dump(data, f)
            


def test_1D():
    def f(x):
        return np.sin(x)*np.exp(-x**2/100)

    inputs = np.random.uniform(-10, 10, size=(25, 1))
    outputs = f(inputs)
    print inputs.shape
    print outputs.shape

    answer = raw_input("Load from file?\n")
    if answer.lower() in ["y", "yes"]:
        fname = raw_input("Filename?\n")
        nn = FeedForwardNetwork.from_file(fname)
    else:
        nn = FeedForwardNetwork([1, 5, 1], hidden_activation=np.tanh, regularization_parameter=0.001, verbose=True)    
        nn.optimize((inputs, outputs), attempts=1, error_determination="cross_entropy")
    
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
        
def test_2D():
    def f(x, y, sigma_x=10, sigma_y=10):
        return np.exp(-x**2/sigma_x**2) * np.exp(-y**2/sigma_y**2)
        
    np.random.seed(0)
        
    sigma_x, sigma_y = 50, 10
        
    x, y = np.linspace(-10, 10, 25), np.linspace(-10, 10, 25)
    
    xg, yg = np.meshgrid(x, y, indexing="xy")
    zg = f(xg, yg, sigma_x, sigma_y)
    
    inputs = np.random.uniform(-10, 10, size=(100, 2))
    outputs = f(inputs[:, 0], inputs[:, 1], sigma_x, sigma_y)[:, None]
    
    nn = FeedForwardNetwork([2, 5, 1], regularization_parameter=0.0001, hidden_activation=np.tanh, verbose=False)
    nn.optimize((inputs, outputs), attempts=10, optimization_method="baisin", error_determination="cross_entropy")
    
    nn_input = np.vstack([xg.flatten(), yg.flatten()]).T
    nn_output = nn.sim(nn_input)
    
    fig, ((ax00, ax01, ax02), (ax10, ax11, ax12)) = plt.subplots(2, 3)
    
    ax00.imshow(zg, extent=(xg.min(), xg.max(), yg.min(), yg.max()))
    ax00.scatter(inputs[:, 0], inputs[:, 1])
    ax00.set_title("Target")
   
    ax01.tricontourf(nn_input[:, 0], nn_input[:, 1], nn_output[:, 0])
    ax01.set_title("Neural network output")
    
    tri = ax02.tricontourf(nn_input[:, 0], nn_input[:, 1], np.abs(nn_output.flatten()-zg.flatten())/zg.flatten())
    ax02.set_title("Relative error")
    fig.colorbar(tri, ax=ax02)
    
    x2 = np.zeros(50)
    y2 = np.linspace(-10, 10, 50)
    z_nn = nn.sim(np.vstack([x2, y2]).T).flatten()
    ax10.plot(y2, f(x2, y2, sigma_x, sigma_y), "g-", y2, z_nn, "r-")
    ax10.text(0.1, 0.9, "x = 0", transform=ax10.transAxes)

    x3 = np.linspace(-10, 10, 50)
    y3 = np.zeros(50)
    z2_nn = nn.sim(np.vstack([x3, y3]).T).flatten()
    ax11.plot(x3, f(x3, y3, sigma_x, sigma_y), "g-", x3, z2_nn, "r-")
    ax11.text(0.1, 0.9, "y = 0", transform=ax11.transAxes)

    plt.show()
    

if __name__ == "__main__":
    default_help = argparse.ArgumentDefaultsHelpFormatter
    parser=argparse.ArgumentParser(description="Neural Network Trainingset Creator", 
                                   formatter_class=default_help
                                   )
    parser.add_argument("--test", default="1d", choices=["1d", "2d"], help="Choose test")
    args = parser.parse_args()
    
    if args.test == "1d":
        test_1D()
    else:
        test_2D()
