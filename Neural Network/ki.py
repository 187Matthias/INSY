import numpy

outputs = []
inputs = []

def sigmoid(x):
    return 1/(1+numpy.e^-x)

def sigmoid_derivative(x):
    return x * (1-x)


class Neuron:
    def __init__(self, num_inputs, activation_function = sigmoid, activation_function_d = sigmoid_derivative):
        self.bias = numpy.random.uniform()
        self.weights = numpy.random(size=num_inputs)
        self.inputs = None
        self.output = None
        self.activation_function = activation_function
        self.activation_function_d = activation_function_d

    def activation(self, inputs):
        self.inputs = inputs
        self.output = self.activation_function(numpy.dot(inputs,self.weights) + self.bias)
        return self.output
    
    def activation_d(self):
        return self.activation_function_d(self.output)
    
    def update_weigths (self, gradiant, learning_rate):
        self.weights += learning_rate * gradiant

    def update_bias (self, gradiant, learning_rate):
        self.bias += learning_rate * gradiant
        
class Layer:
    def __init__(self):
        