import torch

w = torch.tensor(2.,requires_grad=True)
y = 7*w
y.backward()
print('w로 미분한 값:',w.grad)
print()

# backward() 호출시 gradient 는 계속 누적됨
# 누적을 안하기 위해서는 gradient 초기화해야함
# 각각의 값에 대해 gradient 초기화는 아래의 방식으로 w2.grad.zero_()
# 한번에 모든 gradient 초기화는 optimizer에서 zero_grad() 호출하여 초기화
# optim.zero_grad() -> output 생성 -> output.backward() -> optim.step()
w2 = torch.tensor(3.,requires_grad=True)
for epoch in range(5):
    y2 = 5 * w2
    y2.backward() # 미분함수 호출
    print('w2로 미분한 값:',w2.grad)
    w2.grad.zero_() # gradient 초기화

