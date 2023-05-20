import numpy as np
import pytest
from ..helpers.layer import DenseLayer


@pytest.fixture
def dense_layer():
    return DenseLayer(3, 2)


@pytest.fixture
def inputs():
    return np.array([[1, 2, 3], [4, 5, 6]])


def test_dense_layer_forward(dense_layer, inputs):
    dense_layer.forward(inputs)
    assert dense_layer.output.shape == (2, 2)


def test_dense_layer_backward(dense_layer, inputs):
    dense_layer.forward(inputs)
    dvalues = np.ones((2, 2))
    dense_layer.backward(dvalues)

    assert dense_layer.dweights.shape == (3, 2)
    assert dense_layer.dbiases.shape == (1, 2)
    assert dense_layer.dinputs.shape == (2, 3)
