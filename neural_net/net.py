import matplotlib.pyplot as plt
import numpy as np
from .helpers.activation import ActivationReLU, ActivationSoftmax
from .helpers.layer import DenseLayer
from .helpers.loss import CategoricalCrossentropy

import nnfs
from nnfs.datasets import spiral_data
nnfs.init()


def main() -> None:
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

    # Create loss function
    loss_function = CategoricalCrossentropy()

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
    print(softmax_activation.output[:5])

    loss = loss_function.calculate(softmax_activation.output, y)
    print('loss:', loss)


if __name__ == "__main__":
    main()
