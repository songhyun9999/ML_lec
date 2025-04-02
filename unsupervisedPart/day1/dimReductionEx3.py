from sklearn.datasets import load_breast_cancer

cancer = load_breast_cancer()
print(cancer.keys())
print(cancer['feature_names'])
print(len(cancer['feature_names']))
print(cancer.data)

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
plt.rc('font',family='Malgun Gothic')

scaler = StandardScaler()
x_scaled = scaler.fit_transform(cancer.data)

pca = PCA(n_components=2)
x_pca = pca.fit_transform(x_scaled)
print(x_pca.shape)

import mglearn

plt.figure(figsize=(8,8))
mglearn.discrete_scatter(x_pca[:,0],x_pca[:,1],cancer.target)
plt.legend(['악성','양성'],loc='best')
plt.xlabel('첫 번째 주성분')
plt.ylabel('두 번째 주성분')
plt.show()
