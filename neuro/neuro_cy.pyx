#!/usr/bin/python
cimport numpy as np
from scipy.optimize import minimize as minimize
import matplotlib.pylab as plt


def sigmoid(x):
    x*=-1
    np.exp(x, x)
    x+=1
    np.power(x, -1)
    
def identity(x):
    pass


cdef class FeedForwardNetwork:
    
    cdef:
        public double [:] weights
        int [:, ::1] weight_shapes
        int [:, ::1] weight_lengths
        public double [:, ::1] inputs
        public double [:, ::1] outputs
        public double [:, ::1] intermediate
    
    def __cinit__(self, layer_lengths, bias=True, hidden_activation=np.tanh, output_activation=identity, verbose=False):
        """A FeedForwardNetwork object is initialized by providing a list that 
        contains the number of nodes for each layer.
        For example, a FFN object with an input layer with one node, a hidden 
        layer with 5 nodes and an output layer with 2 nodes is initialized via

            ffn = FeedForwardNetwork([1, 5, 2])

        Further (optional) parameters are
          * bias (expects a boolean), which determines whether each layer 
            receives an additional bias node
          * hidden_activation, which determines the activation function for 
            each hidden layer
          * output_activation, which determines the activation function for 
            the output layer"""
        
        self.layer_lengths = layer_lengths
        self.bias = bias
        self.verbose = verbose
        # If bias is activated, the bias weights are left out from the matrix 
        # multiplication
        if self.bias:
            self.weight_slicer = slice(None, -1)
        else:
            self.weight_slicer = slice(None, None)

        self.hidden_activation = hidden_activation
        self.output_activation = output_activation

        weights = []
        weight_shapes = []
        weight_lengths = [0]
        length_counter = 0
        for hl_prev, hl_next in zip(self.layer_lengths[:-1], self.layer_lengths[1:]):
            hl_weight = np.random.random((hl_prev+bias, hl_next))
            weights.append(hl_weight)
            weight_shapes.append(hl_weight.shape)
            length_counter += (hl_prev+bias) * hl_next
        self.weight_shapes = np.array(weight_shapes)
        self.length_counter = np.array(length_counter)
        
        # Flatten all the weights into a single one-dimensional array, 
        # which can then be passed to the scipy optimization routine    
        self.weights = np.hstack([w.flatten() for w in self.weights])

    cdef _forward_prop(self, input_array):
        cdef:
            int i, left_border, right_border
        if self.verbose:
            print "Forward propagation"
        for i in range(self.weight_shapes.shape[0]-1):
            left_border = self.length_counter[i]
            right_border = self.length_counter[i+1]
            weight = self.weights[left_border:right_border].reshape((self.weight_shapes[i, 0], self.weight_shapes[i, 1]))[self.weight_slicer]
            input_array = np.dot(input_array, weight)
            if self.bias:
                input_array += weight[-1]
            self.hidden_activation(input_array)

        weight = self.weights[self.length_counter[-2]:].reshape((self.weight_shapes[-1, 0], self.weight_shapes[-1, 1]))[self.weight_slicer]
        output = np.dot(input_array, weight[self.weight_slicer])
        if self.bias:
            output += weight[-1]
        self.output_activation(output)
        return output

    def sim(self, input_array):
        """Calculates the output for a given array of inputs.
        Expects an array of the shape (No. of inputs, No. of input nodes)"""
        return self._forward_prop(input_array)

    def calc_error(self, inputs, outputs):
        if self.verbose:
            print "Calculate deviation from training set results"
        nn_output = self._forward_prop(inputs)
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
        """Expects a tuple consisting of an array of input values and an array 
        of output values.
        The weights are then optimized until the squared deviation of the neural 
        network's output from the output values becomes minimal."""
        inputs, outputs = training_set
        weights = np.hstack([w.flatten() for w in self.weights])
        results = minimize(fun=self._calc_error, x0=weights, args=(inputs, outputs), method='BFGS')
        result = results["x"]
        print "diff:", (weights - result).sum()
        self.weights = self._unflatten(result)
        return self.weights


def test():
    def f(x):
        return np.sin(x)*np.exp(-x**2/100)
    nn = FeedForwardNetwork([1, 3, 1])
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