import numpy as np
from tensorweaver.layers.layer import Layer
from tensorweaver.parameter import Parameter
from tensorweaver.operators.embedding import embedding


class Embedding(Layer):
    def __init__(self, num_embeddings, embedding_dim, padding_idx=None):
        super().__init__()

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx

        self.embedding_weights = Parameter(
            np.random.randn(num_embeddings, embedding_dim)
        )
        if self.padding_idx is not None:
            self.embedding_weights.data[self.padding_idx] = 0

    def forward(self, input):
        result = embedding(input, self.embedding_weights, self.padding_idx)
        return result
