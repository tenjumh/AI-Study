import numpy as np

class Relu:
    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = (x <= 0)   #mask는 x행렬이 들어오면 x행령에서 0보다 작은 건 True
        out = x.copy()
        out[self.mask] = 0   #x를 copy한 out행렬에 True 위치는 O으로 바꿈

        return out

    def backward(self, dout):
        dout[self.mask] = 0    #
        dx = dout

        return dx
'''
x = np.array([[1.0, -1.5], [-2.0, 3.0]])
print(x)

mask = (x<=0)
print(mask)
test_for = Relu()
a = test_for.forward(x)
print(a)

test_back = Relu()
b = test_back.backward(a)
print(b)
'''
