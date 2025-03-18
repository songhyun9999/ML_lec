from sklearn.datasets import make_regression
import matplotlib.pyplot as plt


features, target, coefficients = make_regression(n_samples=500, n_features=2,
                                                 n_informative=2,n_targets=1,
                                                 noise=0.0, random_state=1,
                                                 coef=True)


# print(features)
# print(target)
# print(coefficients)
print(features.shape)
print(target.shape)

# plt.scatter(features[:,0],features[:,1],s=50,edgecolors='k')
# plt.show()

from sklearn.datasets import make_classification

features, target = make_classification(n_samples=300,n_features=3,
                                       n_informative=3,n_redundant=0,
                                       n_classes=2,
                                       weights=[0.25,0.75],random_state=1)

print(features.shape)
print(target)
print()


from sklearn.datasets import make_blobs

features, target = make_blobs(n_samples=500,
                              n_features=2,
                              centers=4,
                              cluster_std=1,
                              shuffle=True,
                              random_state=1)
print(target)
# plt.scatter(features[:,0],features[:,1],c=target,s=100)
# plt.show()

from sklearn.datasets import make_circles
features, target = make_circles(n_samples=100,
                                factor=0.1,
                                noise=0.1)

# plt.scatter(features[:,0],features[:,1],c=target)
# plt.show()

from sklearn.datasets import make_moons

features, target = make_moons(n_samples=400,noise=0.1,random_state=1)
plt.scatter(features[:,0],features[:,1],c=target,marker='o',s=100)
plt.show()
