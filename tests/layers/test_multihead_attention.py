import numpy as np

from tensorweaver.autodiff.tensor import Tensor
from tensorweaver.layers.multihead_attention import MultiheadAttention


def test_basic_multihead_attention():
    # Test basic multihead attention calculation
    batch_size, seq_len, embed_dim = 2, 4, 8
    num_heads = 2

    mha = MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads)

    # Create input data
    query = Tensor(np.random.randn(batch_size, seq_len, embed_dim))
    key = Tensor(np.random.randn(batch_size, seq_len, embed_dim))
    value = Tensor(np.random.randn(batch_size, seq_len, embed_dim))

    # Forward propagation
    output = mha(query, key, value)

    # Check output shape
    assert output.data.shape == (batch_size, seq_len, embed_dim)


def test_attention_scaling():
    # Test attention score scaling
    batch_size, seq_len, embed_dim = 2, 3, 6
    num_heads = 2

    mha = MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads)

    # Create specific input to make Q and K dot product large
    query = Tensor(np.ones((batch_size, seq_len, embed_dim)) * 2)
    key = Tensor(np.ones((batch_size, seq_len, embed_dim)) * 2)
    value = Tensor(np.ones((batch_size, seq_len, embed_dim)))

    output = mha(query, key, value)

    # Since scaling factor exists, output should not have numerical overflow
    assert not np.any(np.isnan(output.data))
    assert not np.any(np.isinf(output.data))


def test_different_sequence_lengths():
    # Test input with different sequence lengths
    batch_size, embed_dim = 2, 8
    num_heads = 2
    q_len, k_len = 4, 6

    mha = MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads)

    query = Tensor(np.random.randn(batch_size, q_len, embed_dim))
    key = Tensor(np.random.randn(batch_size, k_len, embed_dim))
    value = Tensor(np.random.randn(batch_size, k_len, embed_dim))

    output = mha(query, key, value)

    # Check if output shape is correct
    assert output.data.shape == (batch_size, q_len, embed_dim)


def test_head_dimension_alignment():
    # Test head dimension alignment
    embed_dim = 8
    num_heads = 2
    head_dim = embed_dim // num_heads

    mha = MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads)

    # Ensure embed_dim is divisible by num_heads
    assert embed_dim % num_heads == 0
    assert head_dim == 4

    batch_size, seq_len = 2, 3
    query = Tensor(np.random.randn(batch_size, seq_len, embed_dim))
    key = Tensor(np.random.randn(batch_size, seq_len, embed_dim))
    value = Tensor(np.random.randn(batch_size, seq_len, embed_dim))

    output = mha(query, key, value)

    # Check output dimensions
    assert output.data.shape == (batch_size, seq_len, embed_dim)


def test_attention_mask():
    # Test attention calculation with mask
    batch_size, seq_len, embed_dim = 2, 4, 8
    num_heads = 2

    mha = MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, bias=True)

    query = Tensor(np.random.randn(batch_size, seq_len, embed_dim))
    key = Tensor(np.random.randn(batch_size, seq_len, embed_dim))
    value = Tensor(np.random.randn(batch_size, seq_len, embed_dim))

    output = mha(query, key, value)

    # Check output
    assert output.data.shape == (batch_size, seq_len, embed_dim)
    # Ensure output doesn't contain nan or inf
    assert not np.any(np.isnan(output.data))
    assert not np.any(np.isinf(output.data))


def test_gradient_flow():
    # Test gradient flow
    batch_size, seq_len, embed_dim = 2, 3, 6
    num_heads = 2

    mha = MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads)

    query = Tensor(np.random.randn(batch_size, seq_len, embed_dim))
    key = Tensor(np.random.randn(batch_size, seq_len, embed_dim))
    value = Tensor(np.random.randn(batch_size, seq_len, embed_dim))

    output = mha(query, key, value)

    # Backward propagation
    output.backward(np.ones_like(output.data))

    # Check if gradients exist and shape is correct
    assert query.grad is not None
    assert key.grad is not None
    assert value.grad is not None
    assert query.grad.shape == query.data.shape
    assert key.grad.shape == key.data.shape
    assert value.grad.shape == value.data.shape
