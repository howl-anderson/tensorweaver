from tensorweaver.autodiff.function import Function

class View(Function):
    def __init__(self, *shape):
        super().__init__()
        self.shape = shape
        self.original_shape = None

    def forward(self, input):
        self.original_shape = input.shape
        return input.reshape(*self.shape)
    
    def backward(self, grad_output):
        if grad_output is None:
            return None
        return grad_output.reshape(*self.original_shape)
    
def view(input, *shape):
    return View(*shape)(input)