import numpy as np
import matplotlib.pylab as plt

# 시그모이드 함수
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def ReLU(x):
    return np.maximum(0, x)

if __name__ == '__main__':
    x = np.array([-1.0, 1.0, 2.0])
    print(sigmoid(x))
    t = np.array([1.0, 2.0, 3.0])
    print(1 + t)
    print(1 / t)
    x = np.arange(-5.0, 5.0, 0.01)
    y = sigmoid(x)
    # plt.plot(x, y)
    # plt.ylim(-0.1, 1.1)
    # plt.show()
    y = ReLU(x)
    plt.plot(x, y)
    plt.ylim(-0.1, 5.1)
    plt.show()

