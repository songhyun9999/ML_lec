import torch

### slicing
## slicing 은 numpy 와 동일한 방식으로 사용
t1 = torch.tensor([[1,2,3],[4,5,6]])
print(t1)
print()

print(t1[:,:2])
print()

# boolean indexing
print(t1[t1>4])
print()

print(t1[(t1>4).any(dim=1)])
print()

t1[:,2]=40
print(t1)
print()

t1[t1>4] = 100
print(t1)
print()

t2 = torch.tensor([[1,2,3],[5,6,7]])
t3 = torch.tensor([[8,9,10],[11,22,33]])
print(t2)
print()
print(t3)
print()
t4 = torch.cat([t2,t3],dim=0)
print(t4)
print()

# print(torch.chunk(t4,4,dim=0))
for c in torch.chunk(t4,4,dim=0):
    print(c,end='\n\n')
print()

for c in torch.chunk(t4,3,dim=1):
    print(c,end='\n\n')
print()



