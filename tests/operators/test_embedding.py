import numpy as np

from tensorweaver.autodiff.tensor import Tensor
from tensorweaver.operators.embedding import embedding


def test_basic_embedding():
    # Create a simple embedding weight matrix
    weight = Tensor(
        np.array(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]  # id 0  # id 1  # id 2
        )
    )

    # Create input indices
    indices = Tensor(np.array([1, 0, 2]))

    # Use embedding function
    output = embedding(indices, weight)

    # Expected output should be the corresponding rows
    expected = np.array(
        [[4.0, 5.0, 6.0], [1.0, 2.0, 3.0], [7.0, 8.0, 9.0]]  # row 1  # row 0  # row 2
    )

    assert np.allclose(output.data, expected)


def test_embedding_with_padding():
    weight = Tensor(
        np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])  # id 0  # id 1  # id 2
    )

    indices = Tensor(np.array([1, 0, 2, 0]))

    # Use embedding function with padding_idx
    output = embedding(indices, weight, padding_idx=0)

    # The rows corresponding to padding_idx should remain unchanged
    expected = np.array(
        [
            [3.0, 4.0],  # row 1
            [1.0, 2.0],  # row 0 (padding)
            [5.0, 6.0],  # row 2
            [1.0, 2.0],  # row 0 (padding)
        ]
    )

    assert np.allclose(output.data, expected)


def test_embedding_backward():
    weight = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]))

    indices = Tensor(np.array([0, 1, 0]))

    output = embedding(indices, weight)

    # Simulate gradient from next layer
    grad_output = Tensor(
        np.array(
            [
                [0.1, 0.2],  # grad for first sample
                [0.3, 0.4],  # grad for second sample
                [0.5, 0.6],  # grad for third sample
            ]
        )
    )

    # Backward pass
    output.backward(grad_output)

    # The gradient for each row in weight should be the sum of gradients
    # for all locations where that index was used
    expected_grad = np.array(
        [
            [0.6, 0.8],  # sum of grads for index 0 (first and third sample)
            [0.3, 0.4],  # grads for index 1 (second sample)
        ]
    )

    assert np.allclose(weight.grad.data, expected_grad)


def test_transformer_style_batch_embedding():
    # Create a weight matrix with vocabulary size 4 and embedding dimension 3
    weight = Tensor(
        np.array(
            [
                [1.0, 2.0, 3.0],  # id 0: <PAD>
                [4.0, 5.0, 6.0],  # id 1: token1
                [7.0, 8.0, 9.0],  # id 2: token2
                [10.0, 11.0, 12.0],  # id 3: token3
            ]
        )
    )

    # Create an input with batch_size=2, sequence_length=4
    # Simulating two sequences:
    # Sequence 1: [1, 2, 3, 0]  (0 is padding)
    # Sequence 2: [2, 1, 0, 0]  (the last two 0s are padding)
    indices = Tensor(np.array([[1, 2, 3, 0], [2, 1, 0, 0]]))

    output = embedding(indices, weight, padding_idx=0)

    # Expected output shape should be (batch_size, sequence_length, embedding_dim)
    assert output.data.shape == (2, 4, 3)

    # Verify embeddings for the first sequence are correct
    expected_seq1 = np.array(
        [
            [4.0, 5.0, 6.0],  # token1
            [7.0, 8.0, 9.0],  # token2
            [10.0, 11.0, 12.0],  # token3
            [1.0, 2.0, 3.0],  # padding
        ]
    )

    # Verify embeddings for the second sequence are correct
    expected_seq2 = np.array(
        [
            [7.0, 8.0, 9.0],  # token2
            [4.0, 5.0, 6.0],  # token1
            [1.0, 2.0, 3.0],  # padding
            [1.0, 2.0, 3.0],  # padding
        ]
    )

    assert np.allclose(output.data[0], expected_seq1)
    assert np.allclose(output.data[1], expected_seq2)

    # Test backpropagation
    grad_output = Tensor(np.ones((2, 4, 3)))
    output.backward(grad_output)

    # Verify gradients
    # token0 (padding): used 4 times
    # token1: used 2 times
    # token2: used 2 times
    # token3: used 1 time
    expected_grad = np.array(
        [
            [4.0, 4.0, 4.0],  # gradient for padding
            [2.0, 2.0, 2.0],  # gradient for token1
            [2.0, 2.0, 2.0],  # gradient for token2
            [1.0, 1.0, 1.0],  # gradient for token3
        ]
    )

    # If padding_idx is set, the gradient for padding should be 0
    expected_grad[0] = 0

    assert np.allclose(weight.grad.data, expected_grad)
