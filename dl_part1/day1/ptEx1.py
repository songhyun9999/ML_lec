import torch


t1 = torch.FloatTensor([[1,2],[3,4]])
print(t1)
print(type(t1))
print(t1.size())
print(t1.dtype)
print()

# 주로 이 방식 사용
## numpy arr, list 등 다양한 데이터 사용 가능
t2 = torch.tensor([[1,2],[3,4]],dtype=torch.float32)
print(t2)
print(type(t2))
print(t2.dtype)
print()

print(t2.numpy())
print(type(t2.numpy()))

import numpy as np

ndata = np.array([[1,2,3,4,],[5,6,7,8]],dtype=np.float32)
t3 = torch.from_numpy(ndata)
print(t3)

t4 = torch.tensor(ndata)
print(t4)
