import torch

### tensor 연산 방식
## numpy array 연산과 동일하게 작동함
## 단, dtype 이 일치해야함 -> 다르면 연산불가능
## axis -> dimension
t1 = torch.tensor([1,2,3])
t2 = torch.tensor([5,6,7])
print(t1)
print(t2)

t3 = t1 + 30
print(t3)
print()

t4 = t1+t2
print(t4)
print()

t5 = torch.tensor([[10,20,30],[40,50,60]])
print(t5)
print()

# broadcast 연산 
# t5의 모든 행에 t1을 더함
print(t5+t1)

