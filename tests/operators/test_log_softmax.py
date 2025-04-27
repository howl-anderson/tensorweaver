import numpy as np
import torch

from tensorweaver.autodiff.tensor import Tensor
from tensorweaver.operators.log_softmax import log_softmax


def test_log_softmax_1d():
    # Test 1D input
    x = Tensor(np.array([1.0, 2.0, 3.0]))
    result = log_softmax(x)
    
    # Manually calculate expected result
    x_np = np.array([1.0, 2.0, 3.0])
    expected = x_np - np.max(x_np) - np.log(np.sum(np.exp(x_np - np.max(x_np))))
    
    assert np.allclose(result.data, expected)

def test_log_softmax_2d():
    # Test 2D input, compute along the last dimension
    x = Tensor(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))
    result = log_softmax(x)
    
    # Manually calculate expected result
    x_np = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    max_x = np.max(x_np, axis=-1, keepdims=True)
    exp_x = np.exp(x_np - max_x)
    expected = (x_np - max_x) - np.log(np.sum(exp_x, axis=-1, keepdims=True))
    
    assert np.allclose(result.data, expected)

def test_log_softmax_2d_dim0():
    # Test 2D input, compute along the first dimension
    x = Tensor(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))
    result = log_softmax(x, dim=0)
    
    # Manually calculate expected result
    x_np = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    max_x = np.max(x_np, axis=0, keepdims=True)
    exp_x = np.exp(x_np - max_x)
    expected = (x_np - max_x) - np.log(np.sum(exp_x, axis=0, keepdims=True))
    
    assert np.allclose(result.data, expected)

def test_log_softmax_numerical_stability():
    # Test numerical stability (large numbers)
    x = Tensor(np.array([1000.0, 1000.1, 1000.2]))
    result = log_softmax(x)
    
    # Verify results are in a reasonable range
    assert not np.any(np.isnan(result.data))
    assert not np.any(np.isinf(result.data))
    assert np.all(result.data <= 0)  # log_softmax output is always non-positive
    
    # Test numerical stability (small numbers)
    x = Tensor(np.array([-1000.0, -1000.1, -1000.2]))
    result = log_softmax(x)
    
    assert not np.any(np.isnan(result.data))
    assert not np.any(np.isinf(result.data))
    assert np.all(result.data <= 0)

def test_log_softmax_gradient():
    # Test gradient calculation
    x = Tensor(np.array([1.0, 2.0, 3.0]))
    y = log_softmax(x)
    
    # Create an upstream gradient
    grad_output = np.array([0.1, 0.2, 0.3])
    y.backward(grad_output)
    
    # Verify basic gradient properties
    assert x.grad is not None
    assert x.grad.shape == x.data.shape
    
    # Numerical gradient check
    epsilon = 1e-7
    numerical_grad = np.zeros_like(x.data)
    for i in range(len(x.data)):
        x_plus = x.data.copy()
        x_plus[i] += epsilon
        x_minus = x.data.copy()
        x_minus[i] -= epsilon
        
        # Calculate log_softmax for x_plus
        max_plus = np.max(x_plus)
        exp_plus = np.exp(x_plus - max_plus)
        log_softmax_plus = (x_plus - max_plus) - np.log(np.sum(exp_plus))
        
        # Calculate log_softmax for x_minus
        max_minus = np.max(x_minus)
        exp_minus = np.exp(x_minus - max_minus)
        log_softmax_minus = (x_minus - max_minus) - np.log(np.sum(exp_minus))
        
        # Calculate numerical gradient
        numerical_grad[i] = np.sum((log_softmax_plus - log_softmax_minus) * grad_output) / (2 * epsilon)
    
    # Verify analytical and numerical gradients are close
    assert np.allclose(x.grad, numerical_grad, rtol=1e-5, atol=1e-5)

def test_log_softmax_sum_to_one():
    # Test softmax property: sum of exp(log_softmax) should be 1
    x = Tensor(np.array([1.0, 2.0, 3.0]))
    result = log_softmax(x)
    softmax_result = np.exp(result.data)
    
    assert np.allclose(np.sum(softmax_result), 1.0)

def test_log_softmax_batch():
    # Test batch input
    x = Tensor(np.random.randn(10, 5))  # 10 samples, 5 classes each
    result = log_softmax(x)
    
    # Verify that sum of exp(log_softmax) is 1 for each sample
    softmax_result = np.exp(result.data)
    sums = np.sum(softmax_result, axis=1)
    assert np.allclose(sums, 1.0)

def test_log_softmax_shape_matches_pytorch():
    # Create a 4D tensor, simulating batch image data
    batch_size, channels, height, width = 2, 3, 4, 4
    x_np = np.random.randn(batch_size, channels, height, width)
    
    # TensorWeaver's calculation
    x_tw = Tensor(x_np)
    result_tw = log_softmax(x_tw, dim=1)  # Calculate along channel dimension
    
    # PyTorch's calculation
    x_pt = torch.tensor(x_np)
    result_pt = torch.nn.functional.log_softmax(x_pt, dim=1)
    
    # Check shape and value match
    assert result_tw.data.shape == result_pt.shape
    assert np.allclose(result_tw.data, result_pt.detach().numpy(), rtol=1e-5, atol=1e-5)
    
    # Test 2D case (batch classification problem)
    x_np = np.random.randn(32, 10)  # 32 samples, 10 classes
    x_tw = Tensor(x_np)
    x_pt = torch.tensor(x_np)
    
    result_tw = log_softmax(x_tw, dim=1)
    result_pt = torch.nn.functional.log_softmax(x_pt, dim=1)
    
    assert result_tw.data.shape == result_pt.shape
    assert np.allclose(result_tw.data, result_pt.detach().numpy(), rtol=1e-5, atol=1e-5) 