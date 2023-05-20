import numpy as np
from abc import ABC, abstractmethod


class Loss(ABC):
    def calculate(self, output, y):
        # Calculates the data and regularization losses
        # given model output and ground truth values
        sample_losses = self.forward(output, y)

        # Calculate mean loss
        data_loss = np.mean(sample_losses)

        return data_loss

    @abstractmethod
    def forward(self, y_pred, y_true):
        ...

    @abstractmethod
    def backward(self, dvalues, y_true):
        ...


class CategoricalCrossentropy(Loss):
    def forward(self, y_pred, y_true):
        # Number of samples in a batch
        samples = len(y_pred)
        # Clip data to prevent division by 0
        # Clip both sides to not drag mean towards any value
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        if len(y_true.shape) == 1:
            # Probabilities for target values -
            # only if categorical labels
            correct_confidences = y_pred_clipped[
                range(samples),
                y_true
            ]

        elif len(y_true.shape) == 2:
            # Mask values - only for one-hot encoded labels
            correct_confidences = np.sum(
                y_pred_clipped*y_true,
                axis=1
            )

        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods

    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        labels = len(dvalues[0])

        # If labels are sparse, turn them into one-hot vector
        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]

        # Calculate gradient
        self.dinputs = -y_true / dvalues
        # Normalize gradient
        self.dinputs = self.dinputs / samples
