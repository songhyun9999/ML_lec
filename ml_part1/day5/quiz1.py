from sklearn.datasets import fetch_openml
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

mnist = fetch_openml('mnist_784', as_frame=False)
x, y = mnist.data, mnist.target
print(mnist)

import matplotlib.pyplot as plt
import numpy as np

def plot_digit(image_data):
    image = image_data.reshape(28, 28)
    plt.imshow(image, cmap='binary')
    plt.axis('off')

# some_digit = x[0]
# plot_digit(some_digit)
# plt.show()

# plt.figure(figsize=(9, 9))
# for idx, image_data in enumerate(x[:100]):
#     plt.subplot(10,10, idx+1)
#     plot_digit(image_data)
# plt.subplots_adjust(wspace=0, hspace=0)
# plt.show()

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x[:1000], y[:1000], random_state=42)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV

knn_clf = KNeighborsClassifier()
params = [{'n_neighbors':range(3, 8, 1), 'weights':['uniform', 'distance']}]

grid_search = GridSearchCV(knn_clf, params, cv=5)
grid_search.fit(x, y)
print(grid_search.best_score_)
print(grid_search.best_params_)
print()

bestmodel = grid_search.best_estimator_
bestmodel.fit(x, y)
print(bestmodel.score(x, y))








