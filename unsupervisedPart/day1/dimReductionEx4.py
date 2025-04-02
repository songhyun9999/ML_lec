import mglearn
import matplotlib.pyplot as plt
import numpy as np

plt.rc('font',family='Malgun Gothic')

S = mglearn.datasets.make_signals()
# plt.figure(figsize=(12,3))
# plt.plot(S,'-')
# plt.xlabel('시간')
# plt.ylabel('신호')
# plt.margins(0)
# plt.show()

A = np.random.RandomState(0).uniform(size=(100,3))
X = np.dot(S,A.T)
print('original signal shape : ',S.shape)
print('x shape:',X.shape)

from sklearn.decomposition import PCA
from sklearn.decomposition import NMF

nmf = NMF(n_components=3, random_state=42)
N_S = nmf.fit_transform(X)
print(f'NS shape:{N_S.shape}')

pca = PCA(n_components=3)
P_S = pca.fit_transform(X)

models = [X,S,N_S,P_S]

names = ['노이즈 포함 신호',
         '원본신호',
         'NMF로 복원된 신호',
         'PCA로 복원된 신호']

fig,axes = plt.subplots(4,figsize=(14,6),
                        gridspec_kw={'hspace':0,'wspace':0.5},subplot_kw={'xticks':(),'yticks':()})

for model, name, ax in zip(models,names,axes):
    ax.set_title(name)
    ax.plot(model[:,:3],'-')
    ax.margins(0)
plt.show()