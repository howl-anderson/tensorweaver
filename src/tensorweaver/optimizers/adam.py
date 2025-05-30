import numpy as np
from tensorweaver.optimizers.optimizer import Optimizer


class Adam(Optimizer):
    def __init__(self, params, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        super().__init__(params)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0
        
        # Initialize state for each parameter
        self.state = {}
        for p in self.params:
            self.state[id(p)] = {
                'm': np.zeros_like(p.data),  # First moment
                'v': np.zeros_like(p.data)   # Second moment
            }

    def update_one_parameter(self, p):
        if p.grad is None:
            return
            
        state = self.state[id(p)]
        
        self.t += 1
        
        # Update biased first moment estimate
        state['m'] = self.beta1 * state['m'] + (1 - self.beta1) * p.grad
        # Update biased second raw moment estimate
        state['v'] = self.beta2 * state['v'] + (1 - self.beta2) * (p.grad ** 2)
        
        # Compute bias-corrected first moment estimate
        m_hat = state['m'] / (1 - self.beta1 ** self.t)
        # Compute bias-corrected second raw moment estimate
        v_hat = state['v'] / (1 - self.beta2 ** self.t)
        
        # Update parameters
        p.data = p.data - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)