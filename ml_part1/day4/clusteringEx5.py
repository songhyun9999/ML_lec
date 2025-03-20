from matplotlib.image import imread
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

image =imread('ladybug.png')
print(image.shape)

x = image.reshape(-1,3)
print(x.shape)
# kmeans = KMeans(n_clusters=8, random_state=42).fit(x)
# segmented_img = kmeans.cluster_centers_[kmeans.labels_]
# segmented_img = segmented_img.reshape(image.shape)
# plt.imshow(segmented_img)
# plt.show()

segmented_imgs = []
n_colors = [10,8,6,4,2]
for n_clusters in n_colors:
    kmeans = KMeans(n_clusters=n_clusters,random_state=42).fit(x)
    segmented_img = kmeans.cluster_centers_[kmeans.labels_]
    segmented_imgs.append(segmented_img.reshape(image.shape))

plt.figure(figsize=(8,4))
plt.subplot(231)
plt.imshow(image)
plt.title('Original image')
plt.axis('off')

for idx, n_clusters in enumerate(n_colors):
    plt.subplot(232 + idx)
    plt.imshow(segmented_imgs[idx])
    plt.title(f'{n_clusters} colors')
    plt.axis('off')

plt.show()