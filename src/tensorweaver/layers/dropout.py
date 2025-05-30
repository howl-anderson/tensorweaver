from tensorweaver.layers.layer import Layer
from tensorweaver.operators.dropout import dropout


class Dropout(Layer):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, input):
        if self.training:
            return dropout(input, self.p)
        else:
            return input
