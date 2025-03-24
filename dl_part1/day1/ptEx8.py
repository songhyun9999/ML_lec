import torch

t1 = torch.zeros(1,4)
t2 = torch.zeros(4,1)
print(t1)
print()
print(t2)
print()

print(torch.squeeze(t1))
print(torch.squeeze(t2))
print()

t2 = torch.zeros(2,1,3,1,4)
print(t2.size())
print(torch.squeeze(t2).size())
print(torch.squeeze(t2, dim=3).size())
print()

t3 = torch.zeros(2, 3)
print(t3.size())
print(torch.unsqueeze(t3, dim=0).size())
print(torch.unsqueeze(t3, dim=1).size())
print(torch.unsqueeze(t3, dim=2).size())
