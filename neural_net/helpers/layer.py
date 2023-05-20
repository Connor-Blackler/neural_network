import numpy as np


class DenseLayer:
    def __init__(self, n_inputs, n_neurons):
        # Initialize weights (produces a Gaussian distribution with a mean of 0 and a variance of 1)
        # multiply by 0.01 to generate numbers that are a couple of magnitudes smaller
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases
