import matplotlib.pyplot as plt
import numpy as np

import nnfs
from nnfs.datasets import spiral_data

nnfs.init()


class DenseLayer:
    def __init__(self, n_inputs, n_neurons):
        # Initialize weights (produces a Gaussian distribution with a mean of 0 and a variance of 1)
        # multiply by 0.01 to generate numbers that are a couple of magnitudes smaller
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases


# ReLU activation
class ActivationReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)


# Softmax activation
class ActivationSoftmax:
    def forward(self, inputs):
        # Get unnormalized probabilities
        exp_values = np.exp(inputs - np.max(inputs, axis=1,
                                            keepdims=True))

        # Normalize them for each sample
        probabilities = exp_values / np.sum(exp_values, axis=1,
                                            keepdims=True)
        self.output = probabilities


# Create dataset
X, y = spiral_data(samples=100, classes=3)

# Create Dense layer with 2 input features and 3 output values
dense1 = DenseLayer(2, 3)

# Create ReLU activation (to be used with Dense layer):
relu_activation = ActivationReLU()

# Create second Dense layer with 3 input features (as we take output
# of previous layer here) and 3 output values
dense2 = DenseLayer(3, 3)

# Create Softmax activation (to be used with Dense layer):
softmax_activation = ActivationSoftmax()

# Make a forward pass of our training data through this layer
dense1.forward(X)

# Make a forward pass through activation function
# it takes the output of first dense layer here
relu_activation.forward(dense1.output)

# Make a forward pass through second Dense layer
# it takes outputs of activation function of first layer as inputs
dense2.forward(relu_activation.output)

# Make a forward pass through activation function
# it takes the output of second dense layer here
softmax_activation.forward(dense2.output)

# Let's see output of the first few samples:
print(softmax_activation.output[:5])
