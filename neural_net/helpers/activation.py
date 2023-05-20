import numpy as np
from abc import ABC, abstractmethod


class Activation(ABC):
    @abstractmethod
    def forward(self, inputs):
        ...


class ActivationReLU(Activation):
    # ReLU stands for Rectified Linear Unit. It's a simple activation function that outputs the input
    # directly if it's positive, otherwise, it outputs zero.

    def forward(self, inputs):
        # Store inputs for backpropagation later
        self.inputs = inputs

        # Compute output values from inputs
        # np.maximum(0, inputs) applies the ReLU function to each element in the inputs array.
        # If an input is less than 0, it returns 0. If an input is greater than 0, it returns the input itself.
        self.output = np.maximum(0, inputs)

    def backward(self, dvalues):
        # In the backward pass, we're going through the ReLU function in reverse.
        # This is the part of backpropagation where we calculate the gradients that
        # are fed into the previous layer.

        # Initialize gradients at the start of the backpropagation process
        # Since we need to modify the original variable, let's make a copy of the values first
        self.dinputs = dvalues.copy()

        # When the input values are less than or equal to 0, we set the gradient to 0.
        # This is because the gradient of ReLU for inputs less than 0 is also 0, and for inputs greater than 0, it is 1.
        # So the gradients (dvalues) are just passed to the previous layer in the case of inputs greater than 0,
        # and in case of inputs less than or equal to 0, the gradients are made zero.
        self.dinputs[self.inputs <= 0] = 0


class ActivationSoftmax(Activation):
    """
    This class implements the softmax activation function for a neural network layer.

    The softmax function is often used in the final layer of a neural network
    that needs to output probabilities for multiple classes.
    It turns logits (numeric output of the previous layer) into probabilities that sum to one.
    Softmax function outputs a vector that represents the probability distributions of a list of potential outcomes.

    Attributes
    ----------
    inputs : ndarray
        Input data for the layer, a numpy array of shape (batch_size, input_size).
    output : ndarray
        Output data for the layer, a numpy array of shape (batch_size, output_size).

    Methods
    -------
    forward(inputs):
        Applies the softmax function to the input data.

    backward(dvalues):
        Performs backpropagation for the softmax layer, computing the gradient
        of the loss with respect to the inputs.
    """

    def forward(self, inputs):
        self.inputs = inputs

        # This line calculates the exponential of each input value minus the maximum value in the input.
        # This is a common technique to avoid numerical overflow, a problem that occurs when numbers are
        # too large for the computer's number representation system to handle.
        # The np.max(inputs, axis=1, keepdims=True) expression is finding the maximum value across each row
        # (which corresponds to each sample in the batch) and preserving the array dimensions so that
        # broadcasting works correctly in the subtraction operation.
        exp_values = np.exp(inputs - np.max(inputs, axis=1,
                                            keepdims=True))

        # This line is normalizing these exponential values so that they sum to 1. For each sample,
        # it divides the exponential value of each input by the sum of the exponential values of all inputs.
        # The result is that for each sample, you now have a set of values (probabilities) between 0 and 1
        # that sum to 1. This can be interpreted as the model's "confidence" in each output class.
        probabilities = exp_values / np.sum(exp_values, axis=1,
                                            keepdims=True)

        self.output = probabilities

    def backward(self, dvalues):
        # Initialize an array for the gradients of the inputs
        self.dinputs = np.empty_like(dvalues)

        # For each set of output values and their corresponding gradients from the next layer
        for index, (single_output, single_dvalues) in enumerate(zip(self.output, dvalues)):

            # Reshape the output value to a 2D array
            single_output = single_output.reshape(-1, 1)

            # Calculate the Jacobian matrix of the softmax function.
            # The Jacobian matrix contains the gradients of the softmax function with respect to its inputs.
            # The diagonal of the Jacobian matrix contains the partial derivatives of the softmax output with respect to the corresponding input,
            # and the other entries contain the gradients of the softmax output with respect to the other inputs.
            jacobian_matrix = np.diagflat(
                single_output) - np.dot(single_output, single_output.T)

            # Multiply the Jacobian matrix by the gradients from the next layer (using the chain rule for derivative calculation),
            # sum over the input dimensions (since each neuron's output in the current layer is affected by all neurons' outputs in the previous layer),
            # and store the result as the gradients of the inputs of the current layer.
            self.dinputs[index] = np.dot(jacobian_matrix, single_dvalues)
