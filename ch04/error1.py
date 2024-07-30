import numpy as np
def sum_squares_error(y, t):
    return 0.5 * np.sum((y-t)**2)

# 정답은 '2'
t = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
y = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]
print(sum_squares_error(np.array(y), np.array(t)))      # 0.09750000000000003

def cross_entropy_error(y, t):
    delta = 1e-7
    # 배치 단위가 1일 경우..
    if(y.ndim == 1):
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
    batch_size = y.shape[0]
    return -np.sum(t * np.log(y + delta)) / batch_size


print(cross_entropy_error(np.array(y), np.array(t)))    # 0.510825457099338

