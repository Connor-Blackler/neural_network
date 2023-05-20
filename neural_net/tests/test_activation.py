import numpy as np
import pytest
from ..helpers.activation import ActivationReLU, ActivationSoftmax


@pytest.fixture
def relu():
    return ActivationReLU()


@pytest.fixture
def softmax():
    return ActivationSoftmax()


@pytest.fixture
def inputs():
    return np.array([[-1, 2, -3], [4, -5, 6]])


@pytest.fixture
def dvalues():
    return np.array([[1, 2, 3], [-1, -2, -3]])


def test_relu_forward(relu, inputs):
    relu.forward(inputs)
    assert np.array_equal(relu.output, np.array([[0, 2, 0], [4, 0, 6]]))


def test_relu_backward(relu, inputs, dvalues):
    relu.forward(inputs)
    relu.backward(dvalues)
    assert np.array_equal(relu.dinputs, np.array([[0, 2, 0], [-1, 0, -3]]))


def test_softmax_forward(softmax, inputs):
    softmax.forward(inputs)
    assert np.allclose(np.sum(softmax.output, axis=1), 1)
