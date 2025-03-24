import torch

x1_train = torch.FloatTensor([[73],[93],[89],[96],[73]])
x2_train = torch.FloatTensor([[80],[88],[91],[98],[65]])
x3_train = torch.FloatTensor([[75],[92],[90],[100],[70]])
y_train = torch.FloatTensor([[152],[185],[180],[196],[142]])

w1 = torch.zeros((1,1),requires_grad=True)
w2 = torch.zeros((1,1),requires_grad=True)
w3 = torch.zeros((1,1),requires_grad=True)
b = torch.zeros(1,requires_grad=True)

optimizer = torch.optim.SGD([w1,w2,w3,b],lr=1e-5)

for epoch in range(1000):
    hypothesis = torch.mm(x1_train,w1) + torch.mm(x2_train,w2) + torch.mm(x3_train,w3) + b
    loss = torch.mean((hypothesis - y_train)**2)
    optimizer.zero_grad() # 미분값 초기화
    loss.backward() # loss 미분
    # print(w1.grad)
    optimizer.step() # 미분값으로 weight값 조정

    if epoch % 100 == 0:
        # 값 가져올 때 : scalar -> item(), 배열 -> numpy()
        print(f'epoch:{epoch+1},'
              f' w1:{w1.item()}, w2:{w2.item()}, w3:{w3.item()},'
              f' b:{b.item()}, loss:{loss.item():.3f}')


print()

print(w1,w2,w3)
hypothesis = torch.mm(x1_train, w1) + torch.mm(x2_train, w2) + torch.mm(x3_train, w3) + b
print(hypothesis.detach().numpy())
print(y_train.numpy())