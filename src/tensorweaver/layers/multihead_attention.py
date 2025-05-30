import math
import numpy as np
from tensorweaver.layers.layer import Layer
from tensorweaver.layers.linear import Linear
from tensorweaver.operators.softmax import softmax
from tensorweaver.operators.transpose import transpose
from tensorweaver.autodiff.tensor import Tensor


class MultiheadAttention(Layer):
    def __init__(self, embed_dim, num_heads, dropout=0.0, bias=True):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.bias = bias
        self.head_dim = embed_dim // num_heads

        self.Wq = Linear(embed_dim, embed_dim)
        self.Wk = Linear(embed_dim, embed_dim)
        self.Wv = Linear(embed_dim, embed_dim)
        self.Wout = Linear(embed_dim, embed_dim)

    def bias_mask(self, batch_size, num_heads, q_len, k_len):
        # Create an upper triangular matrix as mask
        mask = np.triu(np.ones((q_len, k_len)), k=1)
        mask = -1e9 * mask
        # Expand dimensions to match the shape of attention scores [batch_size * num_heads, q_len, k_len]
        mask = np.broadcast_to(mask, (batch_size * num_heads, q_len, k_len))
        return Tensor(mask)

    def forward(self, query, key, value):
        batch_size = query.shape[0]
        q_len, k_len = query.shape[1], key.shape[1]

        # Linear transformations
        q = self.Wq(query)
        k = self.Wk(key)
        v = self.Wv(value)

        # Reshape to multi-head form
        q = q.reshape(batch_size, q_len, self.num_heads, self.head_dim)
        k = k.reshape(batch_size, k_len, self.num_heads, self.head_dim)
        v = v.reshape(batch_size, k_len, self.num_heads, self.head_dim)

        # Transpose for attention calculation
        q = transpose(q, 1, 2)  # (batch_size, num_heads, q_len, head_dim)
        k = transpose(k, 1, 2)  # (batch_size, num_heads, k_len, head_dim)
        v = transpose(v, 1, 2)  # (batch_size, num_heads, k_len, head_dim)

        # Reshape for batch matrix multiplication
        q = q.reshape(batch_size * self.num_heads, q_len, self.head_dim)
        k = k.reshape(batch_size * self.num_heads, k_len, self.head_dim)
        v = v.reshape(batch_size * self.num_heads, k_len, self.head_dim)

        # Calculate attention scores
        k_t = transpose(k, 1, 2)  # (batch_size * num_heads, head_dim, k_len)
        scores = q @ k_t  # (batch_size * num_heads, q_len, k_len)
        scores = scores / math.sqrt(self.head_dim)

        if self.bias:
            scores = scores + self.bias_mask(batch_size, self.num_heads, q_len, k_len)

        # Apply softmax
        attn = softmax(scores, dim=-1)

        # Apply attention
        out = attn @ v  # (batch_size * num_heads, q_len, head_dim)

        # Reshape back to original form
        out = out.reshape(batch_size, self.num_heads, q_len, self.head_dim)
        out = transpose(out, 1, 2)  # (batch_size, q_len, num_heads, head_dim)
        out = out.reshape(batch_size, q_len, self.embed_dim)

        # Final linear layer
        out = self.Wout(out)

        return out
