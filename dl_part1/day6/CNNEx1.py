import torch
import torch.nn as nn
import torch.optim as optim


inputs = torch.Tensor(1,1,28,28)
print(inputs.shape)

conv1 = nn.Conv2d(in_channels=1,out_channels=32,kernel_size=3,padding=1,stride=1)
print(conv1)

conv2 = nn.Conv2d(32,64,3,padding=1)
print(conv2)

pool = nn.MaxPool2d(kernel_size=2)
print(pool)
print()


output = conv1(inputs)
print(output.size())

output = conv2(output)
print(output.size())

output = pool(output)
print(output.size())

output = output.view(output.size(0),-1)
print(output.size())

fclayer = nn.Linear(12544,10)
output = fclayer(output)

print(output.size())

