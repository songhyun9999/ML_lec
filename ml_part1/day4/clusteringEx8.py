from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
import mglearn
import matplotlib.pyplot as plt
import numpy as np


x,y = make_moons(n_samples=200,noise=0.05,random_state=0)

scaler = StandardScaler()
scaler.fit(x)
x_scaled = scaler.transform(x)

fig, axes = plt.subplots(1,4,figsize=(15,3),
                         subplot_kw={'xticks':[],'yticks':[]})

algorithms = [KMeans(n_clusters=2),
              AgglomerativeClustering(n_clusters=2),DBSCAN()]

np.random.seed(0)
random_clusters = np.random.randint(low=0,high=2,size=len(x))
axes[0].scatter(x_scaled[:,0],x_scaled[:,1],c=random_clusters,cmap=mglearn.cm3,s=60,edgecolors='black')
axes[0].set_title(f'random-allocation - ARI : {adjusted_rand_score(y,random_clusters):.2f}')

for ax,algorithm in zip(axes[1:],algorithms):
    clusters = algorithm.fit_predict(x_scaled)
    ax.scatter(x_scaled[:,0],x_scaled[:,1],c=clusters,cmap=mglearn.cm3,s=60,edgecolors='k')
    ax.set_title(f'{algorithm.__class__.__name__} - ARI : {adjusted_rand_score(y,clusters):.2f}')

plt.show()

