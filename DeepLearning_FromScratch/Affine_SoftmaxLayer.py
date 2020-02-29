import numpy as np

X = np.random.rand(2)
W = np.random.rand(2, 3)
b = np.random.rand(3)

print(X, X.shape)
print(W, W.shape)
print(b, b.shape)

#Affine Transform : 순전파 때 수행하는 행렬 곱을 기하학에서 "어파인변환"이라 함.
Y = np.dot(X, W) + b
#Y_change = np.dot(W, X) + b
#Y1 = np.matmul(X, W) + b
#Y1_change = np.matmul(W, X) + b

print(Y)


