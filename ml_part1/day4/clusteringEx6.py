import matplotlib.pyplot as plt
import mglearn

# mglearn.plots.plot_agglomerative_algorithm()
# plt.show()

from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering

iris = load_iris()
features = iris.data

scaler = StandardScaler()
features_std = scaler.fit_transform(features)
cluster = AgglomerativeClustering(n_clusters=3,linkage='complete')
model = cluster.fit(features_std)
print(model.labels_)
print()
print(cluster.fit_predict(features_std))
print()
