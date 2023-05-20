import matplotlib.pyplot as plt
import numpy as np
from .helpers.loss_and_activate import ActivationSoftmaxCategoricalCrossentropyLoss
from .helpers.activation import ActivationReLU
from .helpers.layer import DenseLayer

import nnfs
from nnfs.datasets import spiral_data
nnfs.init()


def main() -> None:
    # Create dataset
    X, y = spiral_data(samples=100, classes=3)

    # Create Dense layer with 2 input features and 3 output values
    dense1 = DenseLayer(2, 3)
    dense1.forward(X)

    # Create ReLU activation (to be used with Dense layer):
    activation1 = ActivationReLU()
    activation1.forward(dense1.output)

    # Create second Dense layer with 3 input features (as we take output
    # of previous layer here) and 3 output values (output values)
    dense2 = DenseLayer(3, 3)
    dense2.forward(activation1.output)

    # Create Softmax classifier’s combined loss and activation
    loss_activation = ActivationSoftmaxCategoricalCrossentropyLoss()
    loss = loss_activation.forward(dense2.output, y)

    print(loss_activation.output[:5])
    print('loss:', loss)

    # Calculate accuracy from output of activation2 and targets
    # calculate values along first axis
    predictions = np.argmax(loss_activation.output, axis=1)
    if len(y.shape) == 2:
        y = np.argmax(y, axis=1)
    accuracy = np.mean(predictions == y)

    print('acc:', accuracy)

    # Backward pass
    loss_activation.backward(loss_activation.output, y)
    dense2.backward(loss_activation.dinputs)
    activation1.backward(dense2.dinputs)
    dense1.backward(activation1.dinputs)
    # Print gradients
    print(dense1.dweights)
    print(dense1.dbiases)
    print(dense2.dweights)
    print(dense2.dbiases)


if __name__ == "__main__":
    main()
