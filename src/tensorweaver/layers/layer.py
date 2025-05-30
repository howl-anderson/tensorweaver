from typing import Any

from tensorweaver.parameter import Parameter


class Layer:
    def __init__(self, *args, **kwargs):
        self._parameters = set()
        self._buffers = {}
        self.training = True

    def forward(self, *inputs):
        raise NotImplementedError()

    def __call__(self, *inputs):
        outputs = self.forward(*inputs)
        return outputs

    def __setattr__(self, name: str, value: Any) -> None:
        if isinstance(value, (Parameter, Layer)):
            self._parameters.add(value)
        super().__setattr__(name, value)

    def parameters(self):
        return list(self._get_parameters())

    def _get_parameters(self):
        for p in self._parameters:
            if isinstance(p, Parameter):
                yield p
            elif isinstance(p, Layer):
                yield from p.parameters()

    def register_buffer(self, name: str, tensor):
        self._buffers[name] = tensor
        setattr(self, name, tensor)

    def clean_grad(self):
        for p in self.parameters():
            p.clean_grad()

    def train(self, mode=True):
        """Set the layer to training mode."""
        self.training = mode
        return self

    def eval(self):
        """Set the layer to evaluation mode."""
        return self.train(False)

    def state_dict(self):
        """Mock implementation of state_dict to make it compatible with PyTorch-style saving.

        Returns:
            dict: A dictionary containing a whole state of the module
        """
        state_dict = {}
        # Add parameters
        for p in self._parameters:
            if isinstance(p, Parameter):
                state_dict[id(p)] = p.data
            elif isinstance(p, Layer):
                state_dict.update(p.state_dict())
        # Add buffers
        for name, buf in self._buffers.items():
            state_dict[name] = buf
        return state_dict
