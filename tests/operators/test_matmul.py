import numpy as np
import pytest
from tensorweaver.autodiff.variable import Variable
from tensorweaver.operators.matmul import matmul


def test_matmul():
    # Test basic matrix multiplication
    a = Variable(np.asarray([[1, 2, 3], [4, 5, 6]]))  # 2x3
    b = Variable(np.asarray([[1, 4], [2, 5], [3, 6]]))  # 3x2

    y = matmul(a, b)

    # Verify forward pass
    assert np.array_equal(y.data, np.asarray([[14, 32], [32, 77]]))

    # Test backward pass with gradient of ones
    y.backward(np.ones_like(y.data))

    # Expected gradients:
    # For a: grad = gy @ b.T
    expected_grad_a = np.array([[5, 7, 9], [5, 7, 9]])
    # For b: grad = a.T @ gy
    expected_grad_b = np.array([[5, 5], [7, 7], [9, 9]])

    assert np.array_equal(a.grad, expected_grad_a)
    assert np.array_equal(b.grad, expected_grad_b)


def test_matmul_forward():
    # Test matrix multiplication
    a = Variable(np.array([[1, 2], [3, 4]]))
    b = Variable(np.array([[5, 6], [7, 8]]))
    c = matmul(a, b)
    expected = np.array([[19, 22], [43, 50]])
    assert np.allclose(c.data, expected)
    
    # Test matrix-vector multiplication
    a = Variable(np.array([[1, 2], [3, 4]]))
    b = Variable(np.array([5, 6]))
    c = matmul(a, b)
    expected = np.array([17, 39])
    assert np.allclose(c.data, expected)
    
    # Test batch matrix multiplication
    batch_a = [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]
    batch_b = [[[5, 6], [7, 8]], [[9, 10], [11, 12]]]
    a = Variable(np.array(batch_a))
    b = Variable(np.array(batch_b))
    c = matmul(a, b)
    expected = np.array([[[19, 22], [43, 50]], [[111, 122], [151, 166]]])
    assert np.allclose(c.data, expected)


def test_matmul_backward():
    # Test gradient computation for matrix multiplication
    a = Variable(np.array([[1., 2.], [3., 4.]]))
    b = Variable(np.array([[5., 6.], [7., 8.]]))
    c = matmul(a, b)
    c.backward(np.ones_like(c.data))
    
    # Verify gradients
    assert np.allclose(a.grad, np.array([[11., 15.], [11., 15.]]))
    assert np.allclose(b.grad, np.array([[4., 4.], [6., 6.]]))
    
    # Test gradient computation with scalar gradient
    a = Variable(np.array([[1., 2.], [3., 4.]]))
    b = Variable(np.array([[5., 6.], [7., 8.]]))
    c = matmul(a, b)
    c.backward()  # Uses gradient of ones
    
    assert np.allclose(a.grad, np.array([[11., 15.], [11., 15.]]))
    assert np.allclose(b.grad, np.array([[4., 4.], [6., 6.]]))


def test_matmul_shape_validation():
    # Test invalid shapes
    a = Variable(np.array([[1, 2, 3], [4, 5, 6]]))  # 2x3
    b = Variable(np.array([[1, 2], [3, 4], [5, 6], [7, 8]]))  # 4x2
    
    with pytest.raises(ValueError):
        c = matmul(a, b)  # Should fail: 3 != 4


def test_matmul_vector_cases():
    # Test vector-vector multiplication (inner product)
    a = Variable(np.array([1., 2., 3.]))
    b = Variable(np.array([4., 5., 6.]))
    c = matmul(a, b)
    assert np.isscalar(c.data) or c.data.ndim == 0
    assert np.allclose(c.data, 32.)
    
    # Test gradient for vector-vector case
    c.backward()
    assert np.allclose(a.grad, np.array([4., 5., 6.]))
    assert np.allclose(b.grad, np.array([1., 2., 3.]))
