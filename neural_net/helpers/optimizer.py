import numpy as np


class OptimizerSGD:
    """
    Implements the Stochastic Gradient Descent (SGD) optimization algorithm.

    This class is used to apply the SGD update rule to the weights and biases of a neural network layer.
    SGD uses the gradient of the loss with respect to the weights and biases (computed during backpropagation)
    to update these parameters.

    Attributes
    ----------
    learning_rate : float
        The initial learning rate for the optimizer. Default is 1.
    current_learning_rate : float
        The current learning rate, which may be decayed over time if decay is set.
    decay : float
        The rate at which the learning rate decays over iterations. Default is 0 (no decay).
    iterations : int
        The number of update iterations performed by the optimizer. Default is 0.

    Methods
    -------
    pre_update_params():
        Performs any updates that should happen before the main update step, such as learning rate decay.
    update_params(layer):
        Applies the SGD update rule using the gradients from backpropagation to update the layer's weights and biases.
    post_update_params():
        Performs any updates that should happen after the main update step, such as incrementing the iteration count.
    """

    def __init__(self, learning_rate=1., decay=0.):
        # Initialize optimizer - set settings
        self.learning_rate = learning_rate  # The initial learning rate
        # The current learning rate which can be decayed
        self.current_learning_rate = learning_rate
        self.decay = decay  # Decay rate for the learning rate
        self.iterations = 0  # Number of iterations performed

    def pre_update_params(self):
        # Update the learning rate if decay is set
        if self.decay:
            self.current_learning_rate = self.learning_rate * \
                (1. / (1. + self.decay * self.iterations)
                 )  # Learning rate decay formula

    def update_params(self, layer):
        # Updates parameters using the SGD update rule
        layer.weights += -self.current_learning_rate * layer.dweights  # Update weights
        layer.biases += -self.current_learning_rate * layer.dbiases  # Update biases

    def post_update_params(self):
        # Increment the iterations count after parameters have been updated
        self.iterations += 1


class OptimizerSGDMomentum(OptimizerSGD):
    """
    Stochastic Gradient Descent optimizer with Momentum.

    This optimizer applies the concept of 'momentum' to the base SGD algorithm.
    Momentum helps the gradient descent algorithm to navigate along the relevant directions
    and softens the oscillation in the irrelevant. It does this by adding a fraction of
    the update vector of the past time step to the current update vector.

    Attributes:
        learning_rate: The step size at each iteration while moving toward a minimum of a loss function.
        decay: The decay rate for the learning rate.
        momentum: The momentum factor.
    """

    def __init__(self, learning_rate=1., decay=0., momentum=0.):
        super().__init__(learning_rate, decay)
        # Initialize momentum
        self.momentum = momentum

    def update_params(self, layer):
        """
        Update parameters using SGD with momentum.

        It first checks if momentum terms have been initialized. If they haven't,
        it creates them and sets them to zero. Then it computes the updates for
        weights and biases considering the momentum. Finally, it applies these updates
        to the weights and biases.

        Args:
            layer: The layer for which parameters (weights and biases) need to be updated.
        """

        # Check if previous weight update arrays exist. If they don't, create them filled with zeros
        if not hasattr(layer, 'weight_momentums'):
            layer.weight_momentums = np.zeros_like(layer.weights)
            layer.bias_momentums = np.zeros_like(layer.biases)

        # Compute momentum term and the weight update
        weight_updates = self.momentum * layer.weight_momentums - \
            self.current_learning_rate * layer.dweights
        # Update the layer's weight momentums
        layer.weight_momentums = weight_updates

        # Compute momentum term and the bias update
        bias_updates = self.momentum * layer.bias_momentums - \
            self.current_learning_rate * layer.dbiases
        # Update the layer's bias momentums
        layer.bias_momentums = bias_updates

        # Apply updates to the weights and biases
        layer.weights += weight_updates
        layer.biases += bias_updates
