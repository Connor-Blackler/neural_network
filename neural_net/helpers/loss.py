import numpy as np
from abc import ABC, abstractmethod


class Loss(ABC):
    """
    Base class for loss calculation.

    This abstract class defines the basic structure for calculating loss in
    a neural network. All custom loss classes should inherit from this class
    and implement the `forward` and `backward` methods.

    Methods to implement:
        forward(y_pred, y_true): Calculate the loss.
        backward(dvalues, y_true): Calculate gradient of the loss.

    Method to use:
        calculate(output, y): Calculate the mean loss.

    """

    def calculate(self, output, y):
        """
        Calculate the mean loss.

        This method calculates the mean loss given model output and ground truth values.

        Args:
            output: Predicted values.
            y: True values.

        Returns:
            Mean loss.
        """

        # Calculates the data and regularization losses
        # given model output and ground truth values
        sample_losses = self.forward(output, y)

        # Calculate mean loss
        data_loss = np.mean(sample_losses)

        return data_loss

    @abstractmethod
    def forward(self, y_pred, y_true):
        """
        Forward pass.

        This abstract method calculates the loss given predicted values and true values.
        It needs to be implemented in any child class that inherits from this class.

        Args:
            y_pred: Predicted values.
            y_true: True values.

        Raises:
            NotImplementedError: This method needs to be implemented in the child class.
        """
        pass

    @abstractmethod
    def backward(self, dvalues, y_true):
        """
        Backward pass.

        This abstract method calculates the gradient of the loss with respect to
        predicted values. It needs to be implemented in any child class that inherits
        from this class.

        Args:
            dvalues: Gradient of loss with respect to output of the last layer.
            y_true: True values.

        Raises:
            NotImplementedError: This method needs to be implemented in the child class.
        """
        pass


class CategoricalCrossentropyLoss(Loss):
    """
    Categorical Crossentropy Loss.

    This class calculates the Categorical Crossentropy loss between predicted values
    and true values. This is a common loss function for tasks like multi-class
    classification, where the target values are one-hot encoded.

    Inherits from:
        Loss: A base class for other loss functions.

    Method to implement:
        forward(y_pred, y_true): Calculate the loss.
        backward(dvalues, y_true): Calculate gradient of the loss.
    """

    def forward(self, y_pred, y_true):
        """
        Forward pass.

        Calculates the Categorical Crossentropy loss given predicted values and true values.

        Args:
            y_pred: Predicted values.
            y_true: True values.

        Returns:
            Loss values.
        """

        # Number of samples in a batch
        samples = len(y_pred)

        # Clip data to prevent division by 0
        # Clip both sides to not drag mean towards any value
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        if len(y_true.shape) == 1:
            # Probabilities for target values -
            # only if categorical labels
            correct_confidences = y_pred_clipped[range(samples), y_true]

        elif len(y_true.shape) == 2:
            # Mask values - only for one-hot encoded labels
            correct_confidences = np.sum(y_pred_clipped*y_true, axis=1)

        # Negative log likelihood
        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods

    def backward(self, dvalues, y_true):
        """
        Backward pass.

        Calculates the gradient of the Categorical Crossentropy loss with respect to
        predicted values.

        Args:
            dvalues: Gradient of loss with respect to output of the last layer.
            y_true: True values.

        Returns:
            Gradient of the loss with respect to the input of the layer.
        """

        # Number of samples
        samples = len(dvalues)
        # Number of labels in every sample
        labels = len(dvalues[0])

        # If labels are sparse, turn them into one-hot vector
        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]

        # Calculate gradient
        self.dinputs = -y_true / dvalues
        # Normalize gradient
        self.dinputs = self.dinputs / samples
