import numpy as np

# ReLU 함수 계층
class Relu:
    def __init__(self):
        self.mask = None;   # T/F로 구성된 넘파이 배열

    def forward(self, x):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0
        return out

    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout
        return dx

# 시그모이드 계층
class Sigmoid:
    def __init__(self):
        self.out = None

    def forward(self, x):
        out = 1 / (1 + np.exp(-x))
        self.out = out

    def backward(self, dout):
        dx = dout * (1.0 - self.out) * self.out
        return dx