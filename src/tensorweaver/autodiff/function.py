from typing import List, Union

from numpy.typing import NDArray

from tensorweaver.autodiff.variable import Variable

from typeguard import typechecked


@typechecked
class Function:
    def __init__(self):
        self.inputs: List[Variable] = []
        self.input_data: List[NDArray] = []
        self.outputs: List[Variable] = []
        self.output_data: List[NDArray] = []

    def forward(self, *inputs: NDArray) -> List[NDArray]:
        raise NotImplementedError()

    def backward(self, *inputs: NDArray) -> List[NDArray]:
        raise NotImplementedError()

    def __call__(self, *inputs: Variable) -> Union[Variable, List[Variable]]:
        self.inputs = inputs
        self.input_data = [i.data for i in inputs]

        outputs = self.forward(*self.input_data)

        if not isinstance(outputs, (list, tuple)):
            outputs = (outputs,)

        self.output_data = outputs

        outputs = [Variable(o) for o in outputs]

        for v in outputs:
            v.creator = self

        if not isinstance(outputs, (tuple, list)):
            outputs = (outputs,)

        self.outputs = outputs

        if len(outputs) == 1:
            return outputs[0]
        else:
            return outputs
