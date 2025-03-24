import torch


###### view, reshape 차이점 -> 적용 결과물은 같음
### view
# 메모리를 직접 참조하기 때문에 실행속도가 더 빠름
# 메모리를 직접 참조하기 때문에 contiguous 가 True 인 경우에만 사용 가능
### reshape
# 메모리를 참조하기보단 값을 읽어와 형태를 변환 시킴
# view 보단 느리지만 텐서가 어떤 형태여도 사용 가능함
## contiguous
# True 인 경우 텐서의 값이 연속된 형태로 저장된 것을 의미
# reshape,extend 등의 변형을 거치지 않으면 True임
# 변형을 거치게 되면 tensor.contiguous() 를 통해 강제로 True로 변형시킬 수 있음
# tensor.stride() 를 통해 각 축에 대한 데이터간 메모리 증가 크기를 알 수 있음


t1 = torch.tensor([1,2,3,4,5,6])
print(t1)
print()

t2 = t1.view(2,3)
print(t2)
print()
print(t1.reshape(2,3))
print()

t3 = torch.tensor([[1,2],[3,4],[5,6]])
print(t3)
print(t3.size())
print()
print(t3.view(-1))
print()
print(t3.view(1,-1))
print()
print(t3.view(2,-1))
print()
print(t3.view(3,-1))
print()

t4 = torch.tensor([[[1,2],[4,5]],[[5,6],[7,8]]])
print(t4)
print(t4.size())
print()
print(t4.view(-1))
print()
print(t4.view(1,-1))
print()
print(t4.view(2,-1))
print()
print(t4.view(4,-1))
print()
print(t4.view(2,2,-1))
print()

t5 = torch.tensor([[1,2,3],[4,5,6]])
t6 = torch.tensor([[10,20,30],[40,50,60]])
print(t5)
print()
print(t6)
print()

print(torch.cat([t5,t6],dim=0))
print()
print(torch.cat([t5,t6],dim=1))
print()
