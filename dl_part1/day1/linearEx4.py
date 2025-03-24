import torch

x_train = torch.FloatTensor([[73],[93],[89],[96],[73],
                             [80],[88],[91],[98],[65],
                             [75],[92],[90],[100],[70]]).view(3,5).T
# print(x_train)

y_train = torch.FloatTensor([[152],[185],[180],[196],[142]])

import torch.nn as nn

model = nn.Linear(3,1,bias=True)
optimizer = torch.optim.SGD(model.parameters(),lr=1e-5)
loss_func = nn.MSELoss()
for epoch in range(1000):
    hypothesis = model(x_train)
    loss = loss_func(hypothesis,y_train)
    optimizer.zero_grad() # 미분값 초기화
    loss.backward() # loss 미분
    optimizer.step() # 미분값으로 weight값 조정

    if epoch % 100 == 0:
        # 값 가져올 때 : scalar -> item(), 배열 -> numpy()
        print(f'epoch:{epoch+1}',end='\t')
        for param in model.parameters():
            print(param.detach().numpy(), end='\t')
        print()

print()
