import numpy as np
a = np.array([0.3, 2.9, 4.0])

exp_a = np.exp(a)
print(exp_a)     # [ 1.34985881 18.17414537 54.59815003]

sum_exp_a = np.sum(exp_a)
print(sum_exp_a) # 74.1221542101633

y = exp_a / sum_exp_a
print(y)         # [0.01821127 0.24519181 0.73659691]

# 위의 예시를 함수로 구현.
def softmax(a):
    exp_a = np.exp(a)
    sum_exp_a = np.sum(exp_a)
    return exp_a / sum_exp_a

print(softmax(a)) # [0.01821127 0.24519181 0.73659691]

a = np.array([1010, 1000, 990])
#print(np.exp(a) / np.sum(np.exp(a))) # 제대로 계산되지 않는다.

c = np.max(a) # c = 1010 (최댓값)
print(a - c)  # [  0 -10 -20]
# a-c를 적용한 softmax 함수
print(np.exp(a - c) / np.sum(np.exp(a - c))) # [9.99954600e-01 4.53978686e-05 2.06106005e-09]

print(np.sum(np.exp(a - c)))
print("="*50)
a = np.array([0.3, 2.9, 4.0])
y = softmax(a)
print(y)
print(np.sum(y))