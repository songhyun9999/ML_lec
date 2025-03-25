import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader


x_train = torch.FloatTensor([[73],[93],[89],[96],[73],
                             [80],[88],[91],[98],[65],
                             [75],[92],[90],[100],[70]]).view(3,5).T

y_train = torch.FloatTensor([[152],[185],[180],[196],[142]])

dataset = TensorDataset(x_train,y_train)
dataloader = DataLoader(dataset,batch_size=3,shuffle=True)

class MLRegressionModel(nn.Module):
    def __init__(self):
        super(MLRegressionModel,self).__init__()
        self.linear = nn.Linear(3,1)


    def forward(self,x):
        return self.linear(x)


model = MLRegressionModel()
optimizer = optim.SGD(model.parameters(),lr=1e-5)

for epoch in range(2000):
    for step,batch in enumerate(dataloader):
        x,y = batch
        hypothesis = model(x)
        loss = nn.functional.mse_loss(hypothesis,y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 100 ==0:
            print(f'epoch:{epoch+1} cost{loss.item():.4f}')