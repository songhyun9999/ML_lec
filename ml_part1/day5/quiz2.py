import pandas as pd

passengers = pd.read_csv('titanic_data.csv')
passengers.info()
# print(passengers['Pclass'])
# print()

dummies = pd.get_dummies(passengers['Pclass'])
# print(dummies)
# print()
del passengers['Pclass']
passengers = pd.concat([passengers, dummies], axis=1, join='inner')
passengers.info()
print()

#print(passengers['Sex'])

passengers['Age'].fillna(passengers['Age'].mean(), inplace=True)
passengers.info()

passengers['Sex'] = passengers['Sex'].map({'male':0, 'female':1})
print(passengers['Sex'])

passengers.rename(columns={1:'FirstClass', 2:'SecondClass',3:'EtcClass'}, inplace=True)
passengers.info()

features = passengers[['Sex', 'Age', 'FirstClass', 'SecondClass', 'EtcClass']]
target = passengers[['Survived']]

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(features, target, random_state=42)
print(x_train)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(x_train)
x_train_scaled = scaler.transform(x_train)
x_test_scaled = scaler.transform(x_test)

from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(x_train_scaled, y_train)

print('train data accuracy :',model.score(x_train_scaled, y_train))
print('test data accuracy :',model.score(x_test_scaled, y_test))

import numpy as np

kim = np.array([0.0, 20.0, 0.0, 0.0, 1.0])
hyo = np.array([1.0, 17.0, 1.0, 0.0, 0.0])
choi = np.array([0.0, 32.0, 0.0, 1.0, 0.0])

sample_passengers = np.array([kim, hyo, choi])
sample_passengers_scaled = scaler.transform(sample_passengers)
survive_predict = model.predict(sample_passengers_scaled)
for name, survive in zip(['kim', 'hyo', 'choi'], survive_predict):
    print(f'{name:5} 생존 예측 : {"생존" if survive else "사망"}')















