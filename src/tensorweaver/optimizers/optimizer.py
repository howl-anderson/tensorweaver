class Optimizer:
    def __init__(self, params):
        self.params = params

    def step(self):
        for p in self.params:
            self.update_one_parameter(p)

    def update_one_parameter(self, p):
        raise NotImplementedError()

    def zero_grad(self):
        for p in self.params:
            p.grad = None
