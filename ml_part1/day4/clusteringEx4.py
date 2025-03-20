import pandas as pd

df = pd.DataFrame(columns=['height', 'weight'])
df.loc[0] = [185,60]
df.loc[1] = [180,60]
df.loc[2] = [185,70]
df.loc[3] = [165,63]
df.loc[4] = [155,68]
df.loc[5] = [170,75]
df.loc[6] = [175,80]

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

data_point = df.values
kmeans = KMeans(n_clusters=3).fit(data_point)
print(kmeans.cluster_centers_)
df['cluster_id'] = kmeans.labels_
print(df)

sns.lmplot(x='height', y='weight',
           data=df, fit_reg=False,
           scatter_kws={'s':200},
           hue='cluster_id')
plt.show()











