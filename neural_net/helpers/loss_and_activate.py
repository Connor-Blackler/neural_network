import numpy as np
from .activation import ActivationSoftmax
from .loss import CategoricalCrossentropyLoss


class ActivationSoftmaxCategoricalCrossentropyLoss():
    def __init__(self):
        self.activation = ActivationSoftmax()
        self.loss = CategoricalCrossentropyLoss()

    def forward(self, inputs, y_true):
        self.activation.forward(inputs)
        self.output = self.activation.output

        # Calculate and return loss value
        return self.loss.calculate(self.output, y_true)

    def backward(self, dvalues, y_true):
        samples = len(dvalues)

        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)

        self.dinputs = dvalues.copy()

        self.dinputs[range(samples), y_true] -= 1
        # Normalize gradient
        self.dinputs = self.dinputs / samples
