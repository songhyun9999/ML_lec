from sklearn.datasets import fetch_openml
import numpy as np

mnist = fetch_openml('mnist_784',as_frame=False)
mnist.target = mnist.target.astype(np.uint8)

np.random.seed(42)

m=10000
idx = np.random.permutation(60000)[:m]

x = mnist['data'][idx]
y = mnist['target'][idx]


# t-SNE 를 이용하여 2차원 축소 (특성데이터)
# 2차원 그래프(산점도) 색깔별로 그리기
# target 2,3,5, 데이터를 분리 위 조건처럼 실행

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


tsne = TSNE(n_components=2,random_state=42)
x_reduced = tsne.fit_transform(x)

plt.figure(figsize=(10,8))
plt.scatter(x_reduced[:,0],x_reduced[:,1],c=y,cmap='jet')
plt.axis('off')
plt.colorbar()
plt.show()

import matplotlib as mpl

plt.figure(figsize=(8,8))
cmap = mpl.colormaps.get_cmap('jet')
for digit in (2,3,5):
    plt.scatter(x_reduced[y==digit,0],x_reduced[y==digit,1],c=[cmap(digit/9)])
plt.axis('off')
plt.show()


