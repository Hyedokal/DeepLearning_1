# 곱셈 Layer
class MulLayer:
    def __init__(self):
        self.x = None
        self.y = None

    # 순전파 처리
    def forward(self, x, y):
        self.x = x
        self.y = y
        out = x * y
        return out

    # 역전파 처리
    def backward(self, dout):
        dx = dout * self.y   # x와 y를 바꾼다.
        dy = dout * self.x
        return dx, dy
