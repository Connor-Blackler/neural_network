import matplotlib.pyplot as plt
import numpy as np
from .helpers.loss_and_activate import ActivationSoftmaxCategoricalCrossentropyLoss
from .helpers.activation import ActivationReLU
from .helpers.layer import DenseLayer
from .helpers.optimizer import OptimizerSGD, OptimizerSGDMomentum

import nnfs
from nnfs.datasets import spiral_data
nnfs.init()


def main() -> None:
    # Create dataset
    X, y = spiral_data(samples=100, classes=3)

    # Create Dense layer with 2 input features and 64 output values
    dense1 = DenseLayer(2, 64)
    activation1 = ActivationReLU()
    dense2 = DenseLayer(64, 3)

    # Create Softmax classifier's combined loss and activation
    loss_activation = ActivationSoftmaxCategoricalCrossentropyLoss()

    # Create optimizer
    optimizer = OptimizerSGDMomentum(decay=1e-3, momentum=0.9)

    # Train in loop
    for epoch in range(10001):
        dense1.forward(X)
        activation1.forward(dense1.output)
        dense2.forward(activation1.output)
        loss = loss_activation.forward(dense2.output, y)

        # Calculate accuracy from output of activation2 and targets
        # calculate values along first axis
        predictions = np.argmax(loss_activation.output, axis=1)
        if len(y.shape) == 2:
            y = np.argmax(y, axis=1)
        accuracy = np.mean(predictions == y)

        if not epoch % 100:
            print(f'epoch: {epoch}, ' +
                  f'acc: {accuracy:.3f}, ' +
                  f'loss: {loss:.3f}, ' +
                  f'lr: {optimizer.current_learning_rate}')

        # Backward pass
        loss_activation.backward(loss_activation.output, y)
        dense2.backward(loss_activation.dinputs)
        activation1.backward(dense2.dinputs)
        dense1.backward(activation1.dinputs)

        # Update weights and biases
        optimizer.pre_update_params()
        optimizer.update_params(dense1)
        optimizer.update_params(dense2)
        optimizer.post_update_params()


if __name__ == "__main__":
    main()
