import torch

t1 = torch.linspace(0,3,10)
print(t1)
print()
print(torch.exp(t1))
print(torch.log(t1))
print(torch.cos(t1))
print(torch.sqrt(t1))
print(torch.mean(t1))
print(torch.std(t1))
print()

t2 = torch.tensor([[2,4,6],[7,3,5]])
print(t2)
print()

# dimension 에 따라 함수의 적용이 다름
# dimension 없을 경우 전체 데이터 중 최대값 출력
print(torch.max(t2))
print()
# dimension 0 : 0축(행방향)에 대한 최대 value 텐서와 해당 index 텐서 반환
print(torch.max(t2,dim=0))
print()
# dimension 1 : 1축(열방향)에 대한 최대 value 텐서와 해당 index 텐서 반환
print(torch.max(t2,dim=1))
print()
# 사용 예시
print(torch.max(t2,dim=1)[0]) # 각 행에 대한 최대값 텐서 출력

