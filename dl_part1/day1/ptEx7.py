import torch
import torch.nn.init as init


t1 = init.uniform_(torch.FloatTensor(3,4))
print(t1)
print()

t2 = init.normal_(torch.FloatTensor(3,4),mean=10,std=3)
print(t2)
print()

t3 = torch.FloatTensor(torch.randn(3,4))
print(t3)
print()

t4 = init.constant_(torch.FloatTensor(3,4),val=100)
print(t4)