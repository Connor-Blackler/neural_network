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


X, y = spiral_data(samples=100, classes=3)

dense1 = DenseLayer(2, 3)
dense1.forward(X)
print(dense1.output[:5])
