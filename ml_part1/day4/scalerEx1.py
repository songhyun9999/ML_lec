from sklearn.preprocessing import StandardScaler
import numpy as np

features = np.array([[-500.5],
                     [-100.1],
                     [0],
                     [900.9]])
ss_scaler = StandardScaler()
scaled_features = ss_scaler.fit_transform(features)
print(scaled_features)
print()

from sklearn.preprocessing import RobustScaler
r_scaler = RobustScaler()
scaled_features2 = r_scaler.fit_transform(features)
print(scaled_features2)
print()

from sklearn.preprocessing import MinMaxScaler

minmax_scaler = MinMaxScaler()
scaled_features3 = minmax_scaler.fit_transform(features)
print(scaled_features3)
print()

from sklearn.preprocessing import Normalizer

features = [[0.5,0.5],
            [1.1,3.4],
            [1.5,20.2],
            [1.63,34.4],
            [10.9,3.3]]

n_scaler = Normalizer(norm='l1')
scaled_features4 = n_scaler.fit_transform(features)
print(scaled_features4)