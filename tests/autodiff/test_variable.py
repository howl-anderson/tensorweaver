import numpy as np
import pytest

from tensorweaver.autodiff.variable import Variable
from tensorweaver.autodiff.function import Function


@pytest.mark.parametrize(
    "a,b,expected",
    [
        (Variable(2.0), Variable(3.0), 5.0),
        (2.0, Variable(3.0), 5.0),
        (Variable(2.0), 3.0, 5.0),
    ],
)
def test_add(a, b, expected):
    c = a + b

    assert type(c.data) == np.ndarray
    assert c.data == np.asarray(expected)


@pytest.mark.parametrize(
    "a,b,expected",
    [
        (Variable(2.0), Variable(3.0), -1.0),
        (2.0, Variable(3.0), -1.0),
        (Variable(2.0), 3.0, -1.0),
    ],
)
def test_sub(a, b, expected):
    c = a - b

    assert type(c.data) == np.ndarray
    assert c.data == np.asarray(expected)


@pytest.mark.parametrize(
    "a,b,expected",
    [
        (Variable(2.0), Variable(3.0), 6.0),
        (2.0, Variable(3.0), 6.0),
        (Variable(2.0), 3.0, 6.0),
    ],
)
def test_mul(a, b, expected):
    c = a * b

    assert type(c.data) == np.ndarray
    assert c.data == np.asarray(expected)


@pytest.mark.parametrize(
    "a,b,expected",
    [
        (Variable(12.0), Variable(3.0), 4.0),
        (12.0, Variable(3.0), 4.0),
        (Variable(12.0), 3.0, 4.0),
    ],
)
def test_div(a, b, expected):
    c = a / b

    assert type(c.data) == np.ndarray
    assert c.data == np.asarray(expected)


class AddFunction(Function):
    """A simple add function for testing backward propagation"""
    def forward(self, *inputs):
        x, y = inputs
        return np.array(x) + np.array(y)

    def backward(self, grad):
        return grad, grad


class MulFunction(Function):
    """A simple multiplication function for testing backward propagation"""
    def forward(self, *inputs):
        x, y = inputs
        self._x, self._y = x, y
        return np.array(x) * np.array(y)

    def backward(self, grad):
        return grad * self._y, grad * self._x


def test_backward_simple_add():
    """Test backward propagation through a simple addition"""
    # Create computation graph: z = x + y
    x = Variable(2.0)
    y = Variable(3.0)
    add_fn = AddFunction()
    z = add_fn(x, y)

    # Backward pass
    z.backward()

    # Both inputs should receive gradient 1.0
    assert x.grad == 1.0
    assert y.grad == 1.0


def test_backward_simple_multiply():
    """Test backward propagation through multiplication"""
    # Create computation graph: z = x * y
    x = Variable(2.0)
    y = Variable(3.0)
    mul_fn = MulFunction()
    z = mul_fn(x, y)

    # Backward pass
    z.backward()

    # x's gradient should be y's value and vice versa
    assert x.grad == 3.0
    assert y.grad == 2.0


def test_backward_diamond_shape():
    """Test backward propagation through a diamond-shaped graph"""
    # Create a diamond-shaped graph:
    #     x
    #    / \
    #   +   *
    #    \ /
    #     +
    x = Variable(2.0)
    add_fn1 = AddFunction()
    mul_fn = MulFunction()
    path1 = add_fn1(x, Variable(1.0))  # path1 = x + 1
    path2 = mul_fn(x, Variable(2.0))   # path2 = x * 2
    add_fn2 = AddFunction()
    result = add_fn2(path1, path2)     # result = (x + 1) + (x * 2)

    # Backward pass
    result.backward()

    # x's gradient should be 3.0 (1.0 from addition path + 2.0 from multiplication path)
    assert x.grad == 3.0


def test_backward_clean_grad():
    """Test that clean_grad properly resets gradients"""
    x = Variable(2.0)
    y = Variable(3.0)
    mul_fn = MulFunction()
    z = mul_fn(x, y)

    # First backward pass
    z.backward()
    assert x.grad is not None
    assert y.grad is not None

    # Clean gradients
    x.clean_grad()
    y.clean_grad()
    z.clean_grad()

    assert x.grad is None
    assert y.grad is None
    assert z.grad is None
