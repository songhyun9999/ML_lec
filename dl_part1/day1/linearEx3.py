import torch

x_train = torch.FloatTensor([[73],[93],[89],[96],[73],
                             [80],[88],[91],[98],[65],
                             [75],[92],[90],[100],[70]]).view(3,5).T
# print(x_train)

y_train = torch.FloatTensor([[152],[185],[180],[196],[142]])

W = torch.zeros((3,1),requires_grad=True)
b = torch.zeros(1,requires_grad=True)

optimizer = torch.optim.SGD([W,b],lr=1e-5)
hypothesis = 0
for epoch in range(1000):
    hypothesis = torch.mm(x_train,W) + b
    loss = torch.mean((hypothesis - y_train)**2)
    optimizer.zero_grad() # 미분값 초기화
    loss.backward() # loss 미분
    # print(w1.grad)
    optimizer.step() # 미분값으로 weight값 조정

    if epoch % 100 == 0:
        # 값 가져올 때 : scalar -> item(), 배열 -> numpy()
        print(f'epoch:{epoch+1},'
              f' w1:{W[0].item()}, w2:{W[1].item()}, w3:{W[2].item()},'
              f' b:{b.item()}, loss:{loss.item():.3f}')


print()
print(hypothesis.detach().numpy())
print(y_train.numpy())