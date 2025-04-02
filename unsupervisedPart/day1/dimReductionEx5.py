from sklearn.datasets import make_swiss_roll
import matplotlib.pyplot as plt

X, t = make_swiss_roll(n_samples=1000, noise=0.2, random_state=42)
print(X.shape)
print(t.shape)

axes = [-11.5, 14, -2, 23, -12, 15]

fig = plt.figure(figsize=(6, 5))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=t, cmap=plt.cm.hot)
ax.view_init(10, -70)
ax.set_xlabel("$x_1$", fontsize=18)
ax.set_ylabel("$x_2$", fontsize=18)
ax.set_zlabel("$x_3$", fontsize=18)
ax.set_xlim(axes[0:2])
ax.set_ylim(axes[2:4])
ax.set_zlim(axes[4:6])

plt.show()



#############################################################
from sklearn.decomposition import KernelPCA

lin_pca = KernelPCA(n_components=2, kernel='linear')
rbf_pca = KernelPCA(n_components=2, kernel='rbf', gamma=0.05)
sig_pca = KernelPCA(n_components=2, kernel='sigmoid', coef0=1, gamma=0.001)

#############################################################

plt.figure(figsize=(11, 4))
for subplot, pca, title in ((131, lin_pca, "Linear kernel"), (132, rbf_pca, "RBF kernel, $\gamma=0.04$"), (133, sig_pca, "Sigmoid kernel, $\gamma=10^{-3}, r=1$")):
    X_reduced = pca.fit_transform(X)
    if subplot == 132:
        X_reduced_rbf = X_reduced

    plt.subplot(subplot)
    plt.title(title, fontsize=14)
    plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=t, cmap=plt.cm.hot)
    plt.xlabel("$z_1$", fontsize=18)
    if subplot == 131:
        plt.ylabel("$z_2$", fontsize=18, rotation=0)
    plt.grid(True)

plt.show()


#하이퍼파라미터 튜닝


#############################################################
from sklearn.pipeline import  Pipeline
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.model_selection import GridSearchCV

clf = Pipeline([
    ('kpca', KernelPCA(n_components=2)),
    ('log_reg', LogisticRegression())
])

param_grid = [{
    'kpca__gamma':np.linspace(0.03, 0.05, 10),
    'kpca__kernel':['rbf','sigmoid']
}]

y = t > 6.9
grid_search = GridSearchCV(clf, param_grid, cv=3)
print(grid_search.fit(X, y))
print(grid_search.best_params_)


#############################################################

