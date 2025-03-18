import numpy as np
import matplotlib.pyplot as plt


perch_length = np.array(
    [8.4, 13.7, 15.0, 16.2, 17.4, 18.0, 18.7, 19.0, 19.6, 20.0,
     21.0, 21.0, 21.0, 21.3, 22.0, 22.0, 22.0, 22.0, 22.0, 22.5,
     22.5, 22.7, 23.0, 23.5, 24.0, 24.0, 24.6, 25.0, 25.6, 26.5,
     27.3, 27.5, 27.5, 27.5, 28.0, 28.7, 30.0, 32.8, 34.5, 35.0,
     36.5, 36.0, 37.0, 37.0, 39.0, 39.0, 39.0, 40.0, 40.0, 40.0,
     40.0, 42.0, 43.0, 43.0, 43.5, 44.0])

perch_weight = np.array(
    [5.9, 32.0, 40.0, 51.5, 70.0, 100.0, 78.0, 80.0, 85.0, 85.0,
     110.0, 115.0, 125.0, 130.0, 120.0, 120.0, 130.0, 135.0, 110.0,
     130.0, 150.0, 145.0, 150.0, 170.0, 225.0, 145.0, 188.0, 180.0,
     197.0, 218.0, 300.0, 260.0, 265.0, 250.0, 250.0, 300.0, 320.0,
     514.0, 556.0, 840.0, 685.0, 700.0, 700.0, 690.0, 900.0, 650.0,
     820.0, 850.0, 900.0, 1015.0, 820.0, 1100.0, 1000.0, 1100.0,
     1000.0, 1000.0])

# plt.scatter(perch_length,perch_weight)
# plt.xlabel('length')
# plt.ylabel('weight')
# plt.show()

from sklearn.model_selection import train_test_split
train_x, test_x, train_y, test_y = train_test_split(perch_length,perch_weight,random_state=42,train_size=0.8)
train_x = train_x.reshape(-1,1)
test_x = test_x.reshape(-1,1)
# print(train_x.shape,train_y.shape)
from sklearn.neighbors import KNeighborsRegressor
knr = KNeighborsRegressor()
knr.fit(train_x,train_y)

print(knr.predict(test_x))
print(test_y)
print(f'n_neighbor=5, accuracy = {knr.score(test_x,test_y)}')
print()

knr2 = KNeighborsRegressor(n_neighbors=3)
knr2.fit(train_x,train_y)
print(f'n_neighbor=3, accuracy = {knr2.score(test_x,test_y)}')

knr = KNeighborsRegressor()
x = np.arange(5,45).reshape(-1,1)

plt.figure(figsize=(12,4))
for i,n in enumerate([1,5,10]):
    knr.n_neighbors = n
    knr.fit(train_x,train_y)
    prediction = knr.predict(x)
    plt.subplot(1,3,i+1)
    plt.scatter(train_x,train_y)
    plt.plot(x,prediction)
    plt.xlabel('length')
    plt.ylabel('weight')
    plt.title(f'n_neighbor = {n}')
plt.tight_layout()
plt.show()
