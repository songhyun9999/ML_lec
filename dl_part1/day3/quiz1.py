import torch
import torch.nn as nn
import torch.optim as optim

torch.manual_seed(777) # 재현성을 위해 랜덤값 고정

x_data = torch.FloatTensor([[0,0],[0,1],[1,0],[1,1]])
y_data = torch.FloatTensor([[0],  [1],  [1],  [0]])

model = nn.Sequential(
    nn.Linear(2,10,bias=True),
    nn.Sigmoid(),
    nn.Linear(10, 10),
    nn.Sigmoid(),
    nn.Linear(10,10),
    nn.Sigmoid(),
    nn.Linear(10,1),
    nn.Sigmoid()
)

loss_func = nn.BCELoss()
optimizer = optim.SGD(model.parameters(),lr=1)

for epoch in range(10001):
    hypothesis = model(x_data)
    loss = loss_func(hypothesis,y_data)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print(f'epoch:{epoch}, loss:{loss.item():.4f}')

with torch.no_grad():
    hypothesis = model(x_data)
    prediction = (hypothesis>0.5).float()
    accuracy = (prediction == y_data).float().mean()
    print('hypothesis:\n{}\nnprediction:\n{}\ntarget\n{}\naccuracy:{:.4f}'.format(
        hypothesis.numpy(),prediction.numpy(),y_data.numpy(),accuracy.item()
    ))
