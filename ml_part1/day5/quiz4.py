import pandas as pd
frame = pd.read_csv('Mall_Customers.csv')
frame.info()

from sklearn.preprocessing import StandardScaler
data = frame[['Annual Income (k$)', 'Spending Score (1-100)']]
#print(data)
scaler = StandardScaler()
df_scale = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)
print(df_scale)

from sklearn.cluster import DBSCAN

model = DBSCAN(eps=0.5, min_samples=2)
model.fit(df_scale)
df_scale['cluster'] = model.fit_predict(df_scale)
print(df_scale)

import matplotlib.pyplot as plt
plt.figure(figsize=(8,8))

for i in range(-1, df_scale['cluster'].max() + 1):
    plt.scatter(df_scale.loc[df_scale['cluster'] == i, 'Annual Income (k$)'],
                df_scale.loc[df_scale['cluster'] == i, 'Spending Score (1-100)'],
                label=f'cluster {i}')
plt.legend(loc='best')
plt.title('eps=0.5 min_samples=2', fontsize=19)
plt.xlabel('Annual Income', fontsize=14)
plt.ylabel('Spending Score', fontsize=14)
plt.show()





