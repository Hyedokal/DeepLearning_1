import numpy as np

B = np.array([[1, 2], [3, 4], [5, 6]])
print(B)
print(np.ndim(B))
print(np.shape(B)) # 행렬의 형상을 튜플 형태로 반환 (3행 2열)

A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])
print(np.dot(A, B))
print("="*25)
X = np.array([1, 2])
W = np.array([[1, 3, 5], [2, 4, 6]])
Y = np.dot(X, W)
print(Y) # [5, 11, 17]

def init_network():
	network = {}
	network['name'] = '홍길동'
	return network
nw = init_network()
print(nw)