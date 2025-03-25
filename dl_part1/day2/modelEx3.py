import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.optim as optim


x_train = torch.FloatTensor([[73],[93],[89],[96],[73],
                             [80],[88],[91],[98],[65],
                             [75],[92],[90],[100],[70]]).view(3,5).T

y_train = torch.FloatTensor([[152],[185],[180],[196],[142]])

class CustomDataset(Dataset):
    def __init__(self):
        self.x = torch.FloatTensor([[73],[93],[89],[96],[73],
                             [80],[88],[91],[98],[65],
                             [75],[92],[90],[100],[70]]).view(3,5).T
        self.y = torch.FloatTensor([[152],[185],[180],[196],[142]])

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x = torch.FloatTensor(self.x[idx])
        y = torch.FloatTensor(self.y[idx])

        return x,y


dataset = CustomDataset()
dataloader = DataLoader(dataset,batch_size=2,shuffle=True)

for data in dataloader:
    print(data,end='\n\n')
print()

model = nn.Linear(3,1)
optimizer = optim.SGD(model.parameters(),lr=1e-5)

for epoch in range(20):
    for step,batch in enumerate(dataloader):
        x,y = batch
        hypothesis = model(x)
        loss = nn.functional.mse_loss(hypothesis,y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 100 ==0:
            print(f'epoch:{epoch+1} cost{loss.item():.4f}')