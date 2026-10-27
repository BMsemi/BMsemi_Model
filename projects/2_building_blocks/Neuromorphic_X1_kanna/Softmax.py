import numpy as np
class softmax:
    def __init__(self, temperature=1.0):
        self.temperature = temperature

    def softmax(self,x):
        """Apply softmax to vector x."""
        x = np.array(x,dtype=np.float32)
        z = x - np.max(x)  # for numerical stability
        exp_z = np.exp(z/max(self.temperature,1e-8))
        return exp_z /(np.sum(exp_z)+1e-12)
