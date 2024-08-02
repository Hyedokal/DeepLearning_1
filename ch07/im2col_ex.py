import sys, os
import numpy as np
sys.path.append(os.pardir)
from common.util import im2col

x1 = np.random.rand(1, 3, 7, 7)     # 데이터 수, channel 수, height, width
print(x1)
# filter 크기, stride, padding을 고려하여 입력 데이터를 2차원 배열로 전개한다.
col1 = im2col(x1, 5, 5, stride=1, pad=0) # input_data, filter_h, filter_w, stride, pad
print(col1.shape)       # 9, 75

x2 = np.random.rand(10, 3, 7, 7)
col2 = im2col(x2, 5, 5, stride=1, pad=0)
print(col2.shape)       # 90, 75