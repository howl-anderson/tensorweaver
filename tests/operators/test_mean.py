import numpy as np

from tensorweaver.autodiff.tensor import Tensor
from tensorweaver.operators.mean import mean


def test_mean_forward():
    # Test 1D array
    x = Tensor(np.array([1.0, 2.0, 3.0, 4.0]))
    y = mean(x)
    assert np.allclose(y.data, 2.5)

    # Test 2D array
    x = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]))
    y = mean(x)
    assert np.allclose(y.data, 2.5)

    # Test with axis
    y = mean(x, axis=0)
    assert np.allclose(y.data, np.array([2.0, 3.0]))
    y = mean(x, axis=1)
    assert np.allclose(y.data, np.array([1.5, 3.5]))

    # Test with keepdims
    y = mean(x, axis=0, keepdims=True)
    assert y.data.shape == (1, 2)
    assert np.allclose(y.data, np.array([[2.0, 3.0]]))


def test_mean_backward():
    # Test 1D array
    x = Tensor(np.array([1.0, 2.0, 3.0, 4.0]))
    y = mean(x)
    y.backward()
    # Gradient should be 1/n for each element
    assert np.allclose(x.grad, np.array([0.25, 0.25, 0.25, 0.25]))

    # Test 2D array
    x = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]))
    y = mean(x)
    y.backward()
    # Gradient should be 1/n for each element
    assert np.allclose(x.grad, np.array([[0.25, 0.25], [0.25, 0.25]]))

    # Test with axis
    x = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]))
    y = mean(x, axis=0)
    y.backward(np.array([1.0, 2.0]))
    # Gradient should be 1/n * upstream_grad
    assert np.allclose(x.grad, np.array([[0.5, 1.0], [0.5, 1.0]]))

    # Test with keepdims
    x = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]))
    y = mean(x, axis=0, keepdims=True)
    y.backward(np.array([[1.0, 2.0]]))
    assert np.allclose(x.grad, np.array([[0.5, 1.0], [0.5, 1.0]]))


def test_mean_shape():
    # Test various input shapes
    shapes = [(2, 3), (2, 3, 4), (2, 3, 4, 5)]

    for shape in shapes:
        x = Tensor(np.random.randn(*shape))

        # Test global mean
        y = mean(x)
        assert y.data.shape == ()

        # Test mean along each axis
        for axis in range(len(shape)):
            y = mean(x, axis=axis)
            expected_shape = list(shape)
            expected_shape.pop(axis)
            assert y.data.shape == tuple(expected_shape)

            # Test with keepdims
            y = mean(x, axis=axis, keepdims=True)
            expected_shape = list(shape)
            expected_shape[axis] = 1
            assert y.data.shape == tuple(expected_shape)
