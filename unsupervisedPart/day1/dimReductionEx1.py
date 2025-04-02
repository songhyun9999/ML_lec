import numpy as np

np.random.seed(4)
m = 60
w1, w2 = 0.1, 0.3
noise = 0.1

angles = np.random.rand(m) * 3 * np.pi/2 - 0.5
x = np.empty((m,3))
x[:,0] = np.cos(angles) * np.sin(angles)/2 + noise * np.random.randn(m)/2
x[:,1] = np.sin(angles) * 0.7 + noise * np.random.rand(m)/2
x[:,2] = x[:,0] * w1 + x[:,1]*w2 + noise * np.random.randn(m)
print(x)
print(x.shape)

x_centered = x - x.mean(axis=0)
print(x_centered)
print()

# 공분산 구하기 (Vt)
U, s, Vt = np.linalg.svd(x_centered)

W2 = Vt.T[:,:2]
print(W2)
print()

# 주성분 구하기 (분산큰 2개의 차원)
x2d = x_centered.dot(W2)
print(x2d)
print(x2d.shape)
print()

from sklearn.decomposition import PCA
pca = PCA(n_components=3)
x2d2 = pca.fit_transform(x)
print(x2d2)
print(pca.explained_variance_ratio_) # 각 주성분의 분산 비율 출력
cumsum = np.cumsum(pca.explained_variance_ratio_)
print(cumsum)
d = np.argmax(cumsum >= 0.95) + 1


