import numpy as np

from tensorweaver.autodiff.tensor import Tensor
from tensorweaver.layers.embedding import Embedding


def test_embedding_initialization():
    # Test basic initialization
    num_embeddings = 1000
    embedding_dim = 64
    layer = Embedding(num_embeddings, embedding_dim)

    # Check weight shape
    assert layer.embedding_weights.data.shape == (num_embeddings, embedding_dim)
    # Check whether weight initialization is reasonable (assuming normal distribution initialization)
    assert -3 < layer.embedding_weights.data.mean() < 3
    assert 0 < layer.embedding_weights.data.std() < 2


def test_embedding_forward():
    # Create a small embedding layer for testing
    vocab_size = 10
    embedding_dim = 5
    layer = Embedding(vocab_size, embedding_dim)

    # Set fixed weights for testing
    layer.embedding_weights.data = np.arange(vocab_size * embedding_dim).reshape(
        vocab_size, embedding_dim
    )

    # Test single sequence
    x = Tensor(np.array([1, 3, 5]))
    output = layer(x)
    assert output.data.shape == (3, embedding_dim)
    assert np.allclose(output.data[0], layer.embedding_weights.data[1])
    assert np.allclose(output.data[1], layer.embedding_weights.data[3])
    assert np.allclose(output.data[2], layer.embedding_weights.data[5])

    # Test batch processing
    x_batch = Tensor(np.array([[1, 3, 5], [2, 4, 6]]))
    output_batch = layer(x_batch)
    assert output_batch.data.shape == (2, 3, embedding_dim)


def test_embedding_with_padding():
    vocab_size = 10
    embedding_dim = 5
    padding_idx = 0
    layer = Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)

    # Check whether the embedding at the padding_idx position is 0
    assert np.allclose(layer.embedding_weights.data[padding_idx], 0)

    # Test sequence containing padding
    x = Tensor(np.array([[1, 0, 2], [0, 0, 3]]))
    output = layer(x)

    # Check whether the output at the padding position is 0
    assert np.allclose(output.data[:, 1, :][x.data[:, 1] == padding_idx], 0)


def test_embedding_backward():
    vocab_size = 5
    embedding_dim = 3
    layer = Embedding(vocab_size, embedding_dim)

    # Set fixed weights
    layer.embedding_weights.data = np.array(
        [
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
            [0.7, 0.8, 0.9],
            [1.0, 1.1, 1.2],
            [1.3, 1.4, 1.5],
        ]
    )

    # Forward pass
    x = Tensor(np.array([[0, 1], [2, 3]]))  # batch_size=2, seq_len=2
    output = layer(x)

    # Backward pass
    grad_output = Tensor(np.ones_like(output.data))
    output.backward(grad_output)

    # Check gradient shape
    assert layer.embedding_weights.grad.shape == (vocab_size, embedding_dim)

    # Verify gradient calculation
    # Each word's gradient should be the sum of gradients at its occurrence positions
    expected_grad = np.zeros((vocab_size, embedding_dim))
    for i in range(vocab_size):
        mask = x.data == i
        expected_grad[i] = grad_output.data[mask].sum(axis=0)

    assert np.allclose(layer.embedding_weights.grad.data, expected_grad)


def test_embedding_large_vocab():
    # Test large vocabulary situation
    vocab_size = 50000
    embedding_dim = 768  # BERT-base dimension
    layer = Embedding(vocab_size, embedding_dim)

    # Test large batch input
    batch_size = 32
    seq_length = 512
    x = Tensor(np.random.randint(0, vocab_size, size=(batch_size, seq_length)))

    output = layer(x)
    assert output.data.shape == (batch_size, seq_length, embedding_dim)


def test_embedding_with_zero_init_padding():
    vocab_size = 10
    embedding_dim = 5
    padding_idx = 2
    layer = Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)

    # Verify whether the initialization at the padding position is 0
    assert np.allclose(layer.embedding_weights.data[padding_idx], 0)

    # Forward pass
    x = Tensor(np.array([[1, 2, 3], [2, 2, 1]]))
    output = layer(x)

    # Verify whether the output at the padding position remains 0
    padding_positions = x.data == padding_idx
    assert np.allclose(output.data[padding_positions], 0)
