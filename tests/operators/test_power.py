import numpy as np
import pytest
from tensorweaver.autodiff.tensor import Tensor
from tensorweaver.operators.power import power


def test_power_forward():
    # Test basic power operation
    x = Tensor(np.array([2.0, 3.0]))
    y = Tensor(np.array([2.0, 3.0]))
    z = power(x, y)
    assert np.allclose(z.data, np.array([4.0, 27.0]))

    # Test with scalar exponent
    x = Tensor(np.array([2.0, 3.0]))
    y = Tensor(np.array(2.0))
    z = power(x, y)
    assert np.allclose(z.data, np.array([4.0, 9.0]))

    # Test with integer exponent
    x = Tensor(np.array([2.0, 3.0]))
    y = Tensor(np.array(3))
    z = power(x, y)
    assert np.allclose(z.data, np.array([8.0, 27.0]))


def test_power_backward():
    # Test gradient computation
    x = Tensor(np.array([2.0]))
    y = Tensor(np.array([3.0]))
    z = power(x, y)
    z.backward()

    # dy/dx = 3 * 2^2 = 12
    assert np.allclose(x.grad, np.array([12.0]))
    # dy/dy = 2^3 * ln(2) ≈ 5.545
    assert np.allclose(y.grad, np.array([8.0 * np.log(2.0)]))

    # Test with broadcasting
    x = Tensor(np.array([2.0, 3.0]))
    y = Tensor(np.array(2.0))
    z = power(x, y)
    z.backward(np.array([1.0, 1.0]))
    assert np.allclose(x.grad, np.array([4.0, 6.0]))


def test_power_edge_cases():
    # Test power of 0
    x = Tensor(np.array([0.0]))
    y = Tensor(np.array([2.0]))
    z = power(x, y)
    assert np.allclose(z.data, np.array([0.0]))

    # Test power of 1
    x = Tensor(np.array([1.0]))
    y = Tensor(np.array([5.0]))
    z = power(x, y)
    assert np.allclose(z.data, np.array([1.0]))

    # Test 0th power
    x = Tensor(np.array([2.0]))
    y = Tensor(np.array([0.0]))
    z = power(x, y)
    assert np.allclose(z.data, np.array([1.0]))


def test_power_error_cases():
    # Test power of negative number
    with pytest.raises(ValueError):
        x = Tensor(np.array([0.0]))
        y = Tensor(np.array([-1.0]))
        z = power(x, y)
        z.backward()

    with pytest.raises(ValueError):
        x = Tensor(np.array([0.0]))
        y = Tensor(np.array([0.5]))
        z = power(x, y)
        z.backward()
