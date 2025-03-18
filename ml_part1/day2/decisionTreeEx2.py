import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor

np.random.seed(42)
m = 200
X = np.random.rand(m, 1)
y = 4 * (X - 0.5) ** 2
y = y + np.random.randn(m, 1) / 10

tree_reg1 = DecisionTreeRegressor(random_state=42, max_depth=2)
tree_reg2 = DecisionTreeRegressor(random_state=42, max_depth=3)

# tree_reg1 = DecisionTreeRegressor(random_state=42)
# tree_reg2 = DecisionTreeRegressor(random_state=42, min_samples_leaf=10)

tree_reg1.fit(X,y)
tree_reg2.fit(X,y)

def plot_regression_predictions(tree_reg, X, y, axes=None, ylabel="$y$"):
    if axes is None:
        axes = [0, 1, -0.2, 1]

    x1 = np.linspace(axes[0], axes[1], 500).reshape(-1, 1)
    y_pred = tree_reg.predict(x1)
    # print(y_pred)
    plt.axis(axes)
    plt.xlabel("$x_1$", fontsize=18)
    if ylabel:
        plt.ylabel(ylabel, fontsize=18, rotation=0)
    plt.plot(X, y, "b.")
    plt.plot(x1, y_pred, "r.-", linewidth=2, label=r"$\hat{y}$")


fig, axes = plt.subplots(ncols=2, figsize=(12, 6), sharey=True)
plt.sca(axes[0])
plot_regression_predictions(tree_reg1, X, y)
for split, style in ((0.1973, "k-"), (0.0917, "k--"), (0.7718, "k--")):
    plt.plot([split, split], [-0.2, 1], style, linewidth=2)
plt.text(0.21, 0.65, "Depth=0", fontsize=15)
plt.text(0.01, 0.2, "Depth=1", fontsize=13)
plt.text(0.65, 0.8, "Depth=1", fontsize=13)
plt.legend(loc="upper center", fontsize=18)
plt.title("max_depth=2", fontsize=14)

plt.sca(axes[1])
plot_regression_predictions(tree_reg2, X, y, ylabel=None)
for split, style in ((0.1973, "k-"), (0.0917, "k--"), (0.7718, "k--")):
    plt.plot([split, split], [-0.2, 1], style, linewidth=2)
for split in (0.0458, 0.1298, 0.2873, 0.9040):
    plt.plot([split, split], [-0.2, 1], "k:", linewidth=1)
plt.text(0.3, 0.5, "Depth=2", fontsize=13)
plt.title("max_depth=3", fontsize=14)

plt.show()

# fig, axes = plt.subplots(ncols=2, figsize=(10, 4), sharey=True)
#
# plt.sca(axes[0])
# plt.plot(X, y, "b.")
# plt.plot(x1, y_pred1, "r.-", linewidth=2, label=r"$\hat{y}$")
# plt.axis([0, 1, -0.2, 1.1])
# plt.xlabel("$x_1$", fontsize=18)
# plt.ylabel("$y$", fontsize=18, rotation=0)
# plt.legend(loc="upper center", fontsize=18)
# plt.title("No restrictions", fontsize=14)
#
# plt.sca(axes[1])
# plt.plot(X, y, "b.")
# plt.plot(x1, y_pred2, "r.-", linewidth=2, label=r"$\hat{y}$")
# plt.axis([0, 1, -0.2, 1.1])
# plt.xlabel("$x_1$", fontsize=18)
# plt.title("min_samples_leaf={}".format(tree_reg2.min_samples_leaf), fontsize=14)
# plt.show()
