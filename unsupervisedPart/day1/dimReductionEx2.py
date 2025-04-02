import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA


mnist = fetch_openml('mnist_784',as_frame=False,cache=True)
mnist.target = mnist.target.astype(np.uint8)

X = mnist['data']
Y = mnist['target']
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,random_state=42)

pca = PCA()
pca.fit(X_train)
print(pca.explained_variance_ratio_)

cumsum = np.cumsum(pca.explained_variance_ratio_)
d = np.argmax(cumsum >= 0.95) + 1
print(cumsum)
print(f'dimension:{d}')

# plt.figure(figsize=(6,4))
# plt.plot(cumsum,linewidth=3)
# plt.axis([0,400,0,1])
# plt.xlabel('dimentions')
# plt.ylabel('explained variance')
# plt.grid(True)
# plt.show()

pca = PCA(n_components=154)
X_reduced = pca.fit_transform(X_train)
print(X_reduced.shape)

X_recovered = pca.inverse_transform(X_reduced)


def plot_digits(instances, images_per_row=5, **options):
    size = 28
    images_per_row = min(len(instances), images_per_row)
    images = [instance.reshape(size,size) for instance in instances]
    n_rows = (len(instances) - 1) // images_per_row + 1
    row_images = []
    n_empty = n_rows * images_per_row - len(instances)
    images.append(np.zeros((size, size * n_empty)))
    for row in range(n_rows):
        rimages = images[row * images_per_row : (row + 1) * images_per_row]
        row_images.append(np.concatenate(rimages, axis=1))
    image = np.concatenate(row_images, axis=0)
    plt.imshow(image, cmap = mpl.cm.binary, **options)
    plt.axis("off")

plt.figure(figsize=(7, 4))
plt.subplot(121)
plot_digits(X_train[::2100])
plt.title("Original", fontsize=16)
plt.subplot(122)
plot_digits(X_recovered[::2100])
plt.title("Compressed", fontsize=16)
plt.show()



