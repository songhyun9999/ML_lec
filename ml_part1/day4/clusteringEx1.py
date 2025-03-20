from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

data = load_iris()
print(data.target)
print(data.target_names)

x = data.data

plt.figure(figsize=(9,4))
plt.subplot(121)
plt.scatter(x[:,2],x[:,3],c='k',marker='.')
plt.xlabel('Petal Length',fontsize=14)
plt.ylabel('Petal Width',fontsize=14)
# plt.show()

y=data.target

plt.subplots(122)
plt.plot(x[y==0,2],x[y==0,3],'yo',label='iris setosa')
plt.plot(x[y==1,2],x[y==1,3],'bs',label='iris versicolor')
plt.plot(x[y==2,2],x[y==2,3],'g^',label='iris virginica')
plt.xlabel('Petal Length',fontsize=14)
plt.legend(loc='best')
plt.tick_params(labelleft = False)
plt.show()