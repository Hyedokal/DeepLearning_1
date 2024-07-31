import numpy as np

dY = np.array([[1, 2, 3],[4, 5, 6]])
dB = np.sum(dY, axis=0) # 열끼리 더함
print(dB)               # [5 7 9]

# axis 파라미터의 이해를 위해 넣어 봄
dB = np.sum(dY, axis=1) # 행끼리 더함
print(dB)               # [6 15]