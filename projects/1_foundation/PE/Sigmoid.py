import numpy as np
def sigmoid(x):
    x = np.asarray(x, dtype=np.float32)
    return np.where(x >= 0,
                    1.0 / (1.0 + np.exp(-x)),
                    np.exp(x) / (1.0 + np.exp(x)))