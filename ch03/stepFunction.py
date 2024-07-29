import numpy as np
import matplotlib.pylab as plt

def step_function(x): # 배열은 파라미터의 값이 될 수 없다...
    if(x > 0):
        return 1
    else:
        return 0

def step_function_1(x):
    y = x > 0
    return y.astype(int) # astype(): 데이터의 타입 바꾸는 함수

def step_function_2(x):
    return np.array(x > 0, dtype=int)

if __name__ == '__main__':
    x = np.array([-1.0, 1.0, 2.0])
    print(x)
    y = x > 0
    print(y)
    print(step_function_1(x))

    x = np.arange(-5.0, 5.0, 0.1)   # -5.0 <= x < 5.0 범위, 0.1 간격의 넘파이 배열 생성
    y = step_function_2(x)
    plt.plot(x, y)
    plt.ylim(-0.1, 1.1)  # y축의 범위 지정
    plt.show()