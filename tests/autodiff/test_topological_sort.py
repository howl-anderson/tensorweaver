import numpy as np
from tensorweaver.autodiff.tensor import Tensor
from tensorweaver.autodiff.operator import Operator
from tensorweaver.autodiff.topological_sort import topological_sort


class DummyOperator(Operator):
    """A dummy function for testing topological sort"""

    def forward(self, *inputs):
        # Return a new numpy array instead of the input directly
        return np.array(inputs[0])

    def backward(self, grad):
        return grad


def test_topological_sort_simple_chain():
    """Test topological sort with a simple chain of variables: a -> b -> c"""
    # Create a simple chain: a -> b -> c
    a = Tensor(1.0)
    dummy_fn1 = DummyOperator()
    b = dummy_fn1(a)
    dummy_fn2 = DummyOperator()
    c = dummy_fn2(b)

    # Get topological order starting from c
    sorted_vars = topological_sort(c)

    # Check the order: should be [c, b, a]
    assert len(sorted_vars) == 3
    assert sorted_vars[0] is c
    assert sorted_vars[1] is b
    assert sorted_vars[2] is a


def test_topological_sort_diamond():
    """Test topological sort with a diamond-shaped graph: a -> (b1,b2) -> c"""
    # Create a diamond-shaped graph:
    #     a
    #    / \
    #   b1  b2
    #    \ /
    #     c
    a = Tensor(1.0)
    dummy_fn1 = DummyOperator()
    b1 = dummy_fn1(a)
    dummy_fn2 = DummyOperator()
    b2 = dummy_fn2(a)
    dummy_fn3 = DummyOperator()
    c = dummy_fn3(b1, b2)

    # Get topological order starting from c
    sorted_vars = topological_sort(c)

    # Check the basic properties
    assert len(sorted_vars) == 4
    assert sorted_vars[0] is c  # c should be first
    assert sorted_vars[-1] is a  # a should be last
    # b1 and b2 can be in any order, but should be between c and a
    assert {sorted_vars[1], sorted_vars[2]} == {b1, b2}
