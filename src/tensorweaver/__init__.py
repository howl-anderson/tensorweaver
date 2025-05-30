from contextlib import contextmanager


@contextmanager
def no_grad():
    """Context manager to temporarily disable gradient computation."""
    try:
        # TODO: Actual gradient disabling logic
        yield
    finally:
        # TODO: Restore gradient computation
        pass


from tensorweaver import nn
from tensorweaver import autograd
from tensorweaver import optim
from tensorweaver import onnx

from tensorweaver.autodiff.tensor import Tensor
from tensorweaver.parameter import Parameter


def tensor(data):
    return Tensor(data)


from tensorweaver.operators.faltten import flatten
from tensorweaver.operators.tanh import tanh
from tensorweaver.operators.pow import pow

from tensorweaver.random.manual_seed import manual_seed

from tensorweaver.tensor_create.arange import arange

from tensorweaver.operators.tril import tril

from tensorweaver.tensor_create.ones import ones
from tensorweaver.tensor_create.randint import randint

from tensorweaver.save import save

from tensorweaver.operators.topk import topk
from tensorweaver.operators.lt import lt
from tensorweaver.operators.masked_fill import masked_fill
from tensorweaver.operators.multinomial import multinomial
from tensorweaver.operators.cat import cat
from tensorweaver.operators.softmax import softmax
