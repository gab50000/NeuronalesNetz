import fire
import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(x))

class GRU:
    def __init__(self, vector_size):
        self.Wf = np.random.randn(vector_size, vector_size)
        self.Uf = np.random.randn(vector_size, vector_size)
        self.bf = np.random.randn(vector_size)

        self.Wh = np.random.randn(vector_size, vector_size)
        self.Uh = np.random.randn(vector_size, vector_size)
        self.bh = np.random.randn(vector_size)

        self.ht = np.random.randn(vector_size)

    def __call__(self, x):
        ft = sigmoid(self.Wf @ x + self.Uf @ self.ht + self.bf)
        self.ht = ft * self.ht + (1 - ft) * sigmoid(self.Wh @ x + self.Uh @ (ft * self.ht) + self.bh)
        return self.ht


def test():
    gru = GRU(10)
    print(gru(np.random.random(10)))


fire.Fire()