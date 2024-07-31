import numpy as np
from common.functions import *

class SoftMaxWithLoss:
    def __init__(self):
        self.loss = None # 손실
        self.y = None    # Softmax의 출력
        self.t = None    # 정답 레이블(원-핫 벡터)

    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)
        return self.loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]         # 배치 크기 = 행의 개수
        dx = (self.y - self.t) / batch_size  # 전파값을 배치 수로 나눠, 데이터 1개당 오차를 앞 계층에 전파
        return dx