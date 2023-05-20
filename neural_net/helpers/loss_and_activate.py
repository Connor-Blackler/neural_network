import numpy as np
from .activation import ActivationSoftmax
from .loss import CategoricalCrossentropyLoss


class ActivationSoftmaxCategoricalCrossentropyLoss():
    """
    ActivationSoftmaxCategoricalCrossentropyLoss combines softmax activation function
    and categorical cross-entropy loss function. It is used in multi-class classification tasks.

    Attributes:
        activation (ActivationSoftmax): An instance of softmax activation function.
        loss (CategoricalCrossentropyLoss): An instance of categorical cross-entropy loss function.
        output (np.array): Output from the softmax activation function.
        dinputs (np.array): Gradients to pass to previous layer during backpropagation.
    """

    def __init__(self):
        self.activation = ActivationSoftmax()
        self.loss = CategoricalCrossentropyLoss()

    def forward(self, inputs, y_true):
        """
        Forward pass.

        The input first goes through the softmax activation, and the output of this
        activation function is then used to calculate the loss.

        Args:
            inputs (np.array): Layer inputs.
            y_true (np.array): True target values.

        Returns:
            Loss value.
        """
        self.activation.forward(inputs)
        self.output = self.activation.output

        # Calculate and return loss value
        return self.loss.calculate(self.output, y_true)

    def backward(self, dvalues, y_true):
        """
        Backward pass.

        Calculates gradient of the loss function with respect to the input.

        Args:
            dvalues (np.array): Gradient of loss with respect to output of the last layer.
            y_true (np.array): True target values.
        """

        # Number of samples
        samples = len(dvalues)

        # If targets are one-hot encoded, convert them to discrete values
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)

        self.dinputs = dvalues.copy()
        self.dinputs[range(samples), y_true] -= 1
        # Normalize gradient
        self.dinputs = self.dinputs / samples
