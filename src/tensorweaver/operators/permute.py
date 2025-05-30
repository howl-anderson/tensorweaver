from tensorweaver.autodiff.operator import Operator


class Permute(Operator):
    def __init__(self, dims):
        super().__init__()
        self.dims = dims

    def forward(self, x):
        return x.transpose(self.dims)

    def backward(self, x):
        dim_dict = {v: k for k, v in enumerate(self.dims)}
        reverse_dims = [dim_dict[i] for i in range(len(self.dims))]
        return x.transpose(reverse_dims)


def permute(x, dims):
    return Permute(dims)(x)
