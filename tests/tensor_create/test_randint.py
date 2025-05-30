import numpy as np
import pytest

from tensorweaver.tensor_create.randint import randint


def test_randint_basic():
    # Set seed for reproducibility
    np.random.seed(42)

    # Test basic usage with high parameter only
    output = randint(10, size=(4,))
    assert output.data.shape == (4,)
    assert np.all((output.data >= 0) & (output.data < 10))

    # Test with both low and high
    output = randint(3, 7, size=(4,))
    assert output.data.shape == (4,)
    assert np.all((output.data >= 3) & (output.data < 7))


def test_randint_shapes():
    np.random.seed(42)

    # Test scalar output (no size specified)
    output = randint(10)
    assert output.data.shape == ()
    assert 0 <= output.data < 10

    # Test 2D output
    output = randint(10, size=(2, 3))
    assert output.data.shape == (2, 3)
    assert np.all((output.data >= 0) & (output.data < 10))

    # Test 3D output
    output = randint(10, size=(2, 3, 2))
    assert output.data.shape == (2, 3, 2)
    assert np.all((output.data >= 0) & (output.data < 10))


def test_randint_edge_cases():
    np.random.seed(42)

    # Test with low=high-1 (only one possible value)
    output = randint(5, 6, size=(10,))
    assert np.all(output.data == 5)

    # Test with size=(0,) (empty tensor_create)
    output = randint(0, 10, size=(0,))
    assert output.data.shape == (0,)

    # Test with size=() (scalar)
    output = randint(0, 10, size=())
    assert output.data.shape == ()
    assert 0 <= output.data < 10


def test_randint_distribution():
    np.random.seed(42)

    # Generate large number of samples and check distribution
    size = (1000,)
    low, high = 0, 2
    output = randint(low, high, size=size)

    # Count occurrences of each value
    unique, counts = np.unique(output.data, return_counts=True)

    # Check that we only get values in the correct range
    assert np.all(unique >= low)
    assert np.all(unique < high)

    # Check that we get both possible values
    assert len(unique) == high - low


def test_randint_errors():
    # Test invalid range (high <= low)
    with pytest.raises(ValueError):
        randint(10, 5, size=(4,))

    # Test negative size
    with pytest.raises(ValueError):
        randint(10, size=(-1,))

    # Test invalid high value (non-integer)
    with pytest.raises(TypeError):
        randint(10, 5.5)

    # Test invalid low value (non-integer)
    with pytest.raises(TypeError):
        randint(1.5, 5)


def test_randint_reproducibility():
    # Test that same seed gives same results
    np.random.seed(42)
    output1 = randint(0, 100, size=(5,))

    np.random.seed(42)
    output2 = randint(0, 100, size=(5,))

    np.testing.assert_array_equal(output1.data, output2.data)
