import numpy as np
import matplotlib.pyplot as plt
# 수치 미분 (해석 미분의 근사)
def numerical_diff(f, x):
    # 문제 1. h가 너무 작은 값이여서 반올림 오차 문제를 일으킴.
    # h = 1e-50
    h = 1e-4
    # 문제 2. 차분(임의의 두 점에서 함숫값의 차이)
    # 분자 부분에서도 오차가 생길 수 있음.
    # h를 무한히 0으로 좁히는 것이 불가능해 생기는, 극한을 구현하지 못하는 한계
    # return (f(x+h) - f(x)) / h
    #문제 2 해결: 오차를 줄이기 위해 차분을 개선.
    return (f(x+h) - f(x-h)) / (2*h)

def f(x):
    return 0.01*x**2 + 0.1*x

# 미분한 함수
def tangent_line(f, x):
    d = numerical_diff(f, x)
    print(d)
    y = f(x) - d*x
    return lambda t: d*t + y

x = np.arange(0.0, 20.0, 0.1)
y = f(x)
tf = tangent_line(f, 5)
y2 = tf(x)

plt.xlabel("x")
plt.ylabel("f(x)")
plt.plot(x, y)
plt.plot(x, y2)
plt.show()
# 0.1999999999990898
print(numerical_diff(f, 5))
# 0.2999999999986347
print(numerical_diff(f, 10))