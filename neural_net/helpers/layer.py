import numpy as np


class DenseLayer:
    """
    A dense (fully connected) layer for a neural network.

    This layer performs a dot product of the input and the weights, adds the bias,
    and then passes the result through an activation function.
    """

    def __init__(self, n_inputs, n_neurons):
        """
        Initialize a DenseLayer with random weights and zero biases.

        Parameters:
        - n_inputs: number of inputs to this layer.
        - n_neurons: number of neurons in this layer.
        """
        # Initialize weights with a small random values (Gaussian distribution, mean of 0, variance of 1)
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)

        # Initialize biases as zeroes
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        """
        Perform the forward pass through the layer.

        Parameters:
        - inputs: the input data for the layer.
        """
        # Calculate the dot product of the input and weights and then add biases
        self.output = np.dot(inputs, self.weights) + self.biases

        # Store the input values for use in backpropagation
        self.inputs = inputs

    def backward(self, dvalues):
        """
        Perform the backward pass through the layer, calculating the gradients.

        Parameters:
        - dvalues: the gradient of the loss function with respect to the output of this layer.
        """
        # Gradients on parameters
        self.dweights = np.dot(self.inputs.T, dvalues)

        # Gradient on biases
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)

        # Gradient on values
        self.dinputs = np.dot(dvalues, self.weights.T)
