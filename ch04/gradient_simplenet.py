import sys, os
sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록 설정
import numpy as np
from common.functions import softmax, cross_entropy_error   
from common.gradient import numerical_gradient

class simpleNet:
    def __init__(self):
        self.W = np.random.randn(2,3) # 정규분포로 초기화

    def predict(self, x):
        return np.dot(x, self.W)

    def loss(self, x, t):
        z = self.predict(x)
        y = softmax(z)
        loss = cross_entropy_error(y, t)
        return loss




net = simpleNet()
print(net.W)
x = np.array([0.6, 0.9]) # 입력 데이터
p = net.predict(x)
print(p)
print(np.argmax(p)) # 최댓값의 인덱스
t = np.array([0, 0, 1])  # 정답 레이블
print(net.loss(x, t))

# net.W를 인수로 받아 손실 함수를 계산하는 함수
def f(W):
    return net.loss(x, t)
dW = numerical_gradient(f, net.W)
# 각 행렬의 원소 값은 각 행렬 위치의 가중치를 h만큼 늘렸을 때 손실 함수의 증가량이다.
print(dW)