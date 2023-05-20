
import numpy as np
import pytest
from ..helpers.loss import CategoricalCrossentropyLoss


class TestCategoricalCrossentropyLoss:
    @pytest.fixture
    def loss(self):
        return CategoricalCrossentropyLoss()

    @pytest.fixture
    def y_pred(self):
        return np.array([[0.1, 0.5, 0.4], [0.7, 0.2, 0.1]])

    @pytest.fixture
    def y_true(self):
        return np.array([[0, 1, 0], [1, 0, 0]])

    def test_forward(self, loss, y_pred, y_true):
        output = loss.forward(y_pred, y_true)
        expected_output = np.array([0.69314718, 0.35667494])
        assert np.allclose(
            output, expected_output), f'Expected {expected_output} but got {output}'

    def test_backward(self, loss, y_pred, y_true):
        loss.forward(y_pred, y_true)  # loss forward pass to set internal state
        # assuming dvalues are ones for this test
        dvalues = np.ones(shape=y_pred.shape)
        loss.backward(dvalues, y_true)
        expected_dinputs = np.array([[0., -0.5, 0.], [-0.5, 0., 0.]])
        assert np.allclose(
            loss.dinputs, expected_dinputs), f'Expected {expected_dinputs} but got {loss.dinputs}'
