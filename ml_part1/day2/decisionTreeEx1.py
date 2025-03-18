import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


def plot_decision_boundary(clf, X, y, axes=None, iris=True, legend=False, plot_training=True):
    if axes is None:
        axes = [0, 7.5, 0, 3]
    x1s = np.linspace(axes[0], axes[1], 100)
    x2s = np.linspace(axes[2], axes[3], 100)
    x1, x2 = np.meshgrid(x1s, x2s)
    X_new = np.c_[x1.ravel(), x2.ravel()]
    y_pred = clf.predict(X_new).reshape(x1.shape)
    custom_cmap = ListedColormap(['#fafab0','#9898ff','#a0faa0'])
    plt.contourf(x1, x2, y_pred, alpha=0.3, cmap=custom_cmap)
    if not iris:
        custom_cmap2 = ListedColormap(['#7d7d58','#4c4c7f','#507d50'])
        plt.contour(x1, x2, y_pred, cmap=custom_cmap2, alpha=0.8)
    if plot_training:
        plt.plot(X[:, 0][y==0], X[:, 1][y==0], "yo", label="Iris setosa")
        plt.plot(X[:, 0][y==1], X[:, 1][y==1], "bs", label="Iris versicolor")
        plt.plot(X[:, 0][y==2], X[:, 1][y==2], "g^", label="Iris virginica")
        plt.axis(axes)
    if iris:
        plt.xlabel("Petal length", fontsize=14)
        plt.ylabel("Petal width", fontsize=14)
    else:
        plt.xlabel(r"$x_1$", fontsize=18)
        plt.ylabel(r"$x_2$", fontsize=18, rotation=0)
    if legend:
        plt.legend(loc="lower right", fontsize=14)


from sklearn.datasets import load_iris

# iris = load_iris()
# print(iris.keys())
# print(iris.data[:10])
# print(iris.DESCR)
# print(iris.target)
# print(iris.target_names)
# print(iris.feature_names)

# x = iris.data[:,2:]
# y = iris.target

from sklearn.tree import DecisionTreeClassifier
#
# tree_clf = DecisionTreeClassifier(max_depth=2, random_state=42)
# tree_clf.fit(x,y)
#
# plt.figure(figsize=(8,4))
# plot_decision_boundary(tree_clf,x,y)
# plt.plot([2.45,2.45],[0,3],'k-',linewidth=2)
# plt.plot([2.45,7.5],[1.75,1.75],'k-',linewidth=2)
# plt.plot([4.95,4.95],[0,1.75],'k-',linewidth=2)
# plt.plot([4.85,4.85],[1.75,3],'k-',linewidth=2)
#
# plt.text(1.4,1,'Depth=0',fontsize=15)
# plt.text(3.2,1.8,'Depth=1',fontsize=13)
# plt.text(4.05,0.5,'Depth=2',fontsize=11)
# plt.show()

from sklearn.datasets import make_moons

xm, ym = make_moons(n_samples=100, noise=0.25, random_state=42)

dtree_clf1 = DecisionTreeClassifier(random_state=42)
dtree_clf2 = DecisionTreeClassifier(min_samples_leaf=4, random_state=42)

dtree_clf1.fit(xm,ym)
dtree_clf2.fit(xm,ym)

fig, axes = plt.subplots(ncols=2, figsize=(10,4))
plt.sca(axes[0])

plot_decision_boundary(dtree_clf1,xm,ym,axes=[-1.5,2.4,-1,1.5],iris=False)
plt.title('no restrictions',fontsize=16)

plt.sca(axes[1])

plot_decision_boundary(dtree_clf2,xm,ym,axes=[-1.5,2.4,-1,1.5],iris=False)
plt.title(f'min_samples_leaf={dtree_clf2.min_samples_leaf}',fontsize=16)
plt.ylabel('')
plt.show()





