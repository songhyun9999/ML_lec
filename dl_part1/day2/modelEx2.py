import torch


x_train = torch.FloatTensor([[73],[93],[89],[96],[73],
                             [80],[88],[91],[98],[65],
                             [75],[92],[90],[100],[70]]).view(3,5).T

y_train = torch.FloatTensor([[152],[185],[180],[196],[142]])

from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

dataset = TensorDataset(x_train,y_train)
dataloader = DataLoader(dataset,batch_size=2,shuffle=True)
print(dataloader)
print()

for data in dataloader:
    print(data,end='\n\n')
print()

import torch.nn as nn
import torch.optim as optim

model = nn.Linear(3,1,bias=True)
optimizer = optim.SGD(model.parameters(),lr=1e-5)

for epoch in range(5):
    for step,batch in enumerate(dataloader):
        x,y = batch
        hypothesis = model(x)
        loss = nn.functional.mse_loss(hypothesis,y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f'{epoch+1} / {step+1}, loss:{loss.item():.4f}')



