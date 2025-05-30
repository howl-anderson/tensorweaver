from tensorweaver.layers.layer import Layer
from tensorweaver.operators.sqrt import sqrt


def mean_squared_error(y_true, y_pred):
    return sqrt(y_true - y_pred).mean()


class MSELoss(Layer):
    def forward(self, y_true, y_pred):
        return sqrt(y_true - y_pred).mean()
