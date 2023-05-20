import pytest
import numpy as np
from ..helpers.optimizer import OptimizerSGD, OptimizerSGDMomentum


class LayerMock:
    def __init__(self):
        self.weights = np.array([[1., 1.], [1., 1.]])
        self.biases = np.array([1., 1.])
        self.dweights = np.zeros_like(self.weights)
        self.dbiases = np.zeros_like(self.biases)


def test_learning_rate_decay():
    optimizer = OptimizerSGD(learning_rate=1.0, decay=0.1)
    optimizer.pre_update_params()
    assert optimizer.current_learning_rate == pytest.approx(1.0)
    optimizer.post_update_params()
    optimizer.pre_update_params()
    assert optimizer.current_learning_rate == pytest.approx(0.9090909090909091)


def test_update_params():
    layer = LayerMock()
    layer.weights = np.array([[1.0, 1.0], [1.0, 1.0]])
    layer.biases = np.array([1.0, 1.0])
    layer.dweights = np.array([[0.1, 0.1], [0.1, 0.1]])
    layer.dbiases = np.array([0.1, 0.1])
    optimizer = OptimizerSGD(learning_rate=1.0)
    optimizer.update_params(layer)
    assert np.allclose(layer.weights, np.array([[0.9, 0.9], [0.9, 0.9]]))
    assert np.allclose(layer.biases, np.array([0.9, 0.9]))


def test_momentum_initialization():
    layer = LayerMock()
    optimizer = OptimizerSGDMomentum(
        learning_rate=1.0, decay=0.0, momentum=0.9)
    optimizer.update_params(layer)
    assert hasattr(layer, 'weight_momentums')
    assert hasattr(layer, 'bias_momentums')
    assert np.allclose(layer.weight_momentums, np.zeros_like(layer.weights))
    assert np.allclose(layer.bias_momentums, np.zeros_like(layer.biases))


def test_momentum_update_params():
    layer = LayerMock()
    layer.dweights = np.array([[0.1, 0.1], [0.1, 0.1]])
    layer.dbiases = np.array([0.1, 0.1])
    optimizer = OptimizerSGDMomentum(learning_rate=1.0, momentum=0.9)
    optimizer.update_params(layer)
    assert np.allclose(layer.weights, np.array([[0.9, 0.9], [0.9, 0.9]]))
    assert np.allclose(layer.biases, np.array([0.9, 0.9]))
    assert np.allclose(layer.weight_momentums,
                       np.array([[-0.1, -0.1], [-0.1, -0.1]]))
    assert np.allclose(layer.bias_momentums, np.array([-0.1, -0.1]))
