import numpy as np

class Dropout:
    def __init__(self, dropout_ratio=0.5):
        self.dropout_ratio = dropout_ratio
        self.mask = None

    def forward(self, x, train_flg=True):
        if train_flg:
            # *x.shape: x의 차원을 풀어 각 차원 값으로 파라미터 전달.
            # ex) x가 100x50 행렬이면 np.random.rand(100, 50)과 같은 역할.
            self.mask = np.random.rand(*x.shape) > self.dropout_ratio
            return x * self.mask
        else:
            return x * (1.0 - self.dropout_ratio)

    def backward(self, dout):
        return dout * self.mask
    