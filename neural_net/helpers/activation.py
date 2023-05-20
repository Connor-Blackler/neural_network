import numpy as np
from abc import ABC, abstractmethod


class Activation(ABC):
    @abstractmethod
    def forward(self, inputs):
        ...


class ActivationReLU(Activation):
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)


class ActivationSoftmax(Activation):
    """
    Used to convert to probability outputs, generally used in the output layer
    """

    def forward(self, inputs):
        # Get unnormalized probabilities
        exp_values = np.exp(inputs - np.max(inputs, axis=1,
                                            keepdims=True))

        # Normalize them for each sample
        probabilities = exp_values / np.sum(exp_values, axis=1,
                                            keepdims=True)
        self.output = probabilities
