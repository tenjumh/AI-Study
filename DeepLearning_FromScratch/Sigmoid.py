import numpy as np

class Sigmoid:
    def __init__(self):
        self.out = None

    def forward(self, x):
        out = 1 / (1 + np.exp(x))
        self.out = out

        return out

    def backward(self, dout):
        dx = dout * (1.0 - self.out) * self.out

        return dx
'''
x = np.array([[1.0, -1.5], [-2.0, 3.0]])
print(x)
test_for = Sigmoid()
test_back = Sigmoid()

a = test_for.forward(x)
print(a)
'''
