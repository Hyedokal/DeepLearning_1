import numpy as np
from common.util import im2col

class Convolution:
    def __init__(self, W, b, stride=1, pad=0):
        self.W = W
        self.b = b
        self.stride = stride
        self.pad = pad

    def forward(self, x):
        FN, C, FH, FW = self.W.shape
        N, C, H, W = x.shape
        out_h = int(1 + (H + 2*self.pad - FH) / self.stride)
        out_w = int(1 + (W + 2*self.pad - FW) / self.stride)

        # 입력 데이터를 im2col 함수를 통해 전개
        col = im2col(x, FH, FW, self.stride, self.pad)
        # filter도 reshape를 사용해 2차원 배열로 전개
        # reshape의 두 번째 param으로 -1을 지정하면
        # 다차원 배열의 원소 수가 변환 후에도 똑같이 유지되도록 적절히 묶어준다.
        # 예를 들어 원소 수가 750인 다차원 배열에서 reshape(10, -1)을 호출하면 (10, 75)로 만들어준다.
        col_W = self.W.reshape(FN, -1).T    # 필터 전개
        out = np.dot(col, col_W) + self.b

        out = out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2)
        return out