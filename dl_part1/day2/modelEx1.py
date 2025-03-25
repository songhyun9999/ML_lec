import torch
import torch.nn as nn

x = torch.FloatTensor(torch.randn(16,10))

class CustomLinear(nn.Module):
    def __init__(self, input_size, output_size):
        super(CustomLinear,self).__init__()
        self.linear = nn.Linear(input_size,output_size)

    def forward(self,x):
        y = self.linear(x)
        return y

CModel = CustomLinear(10,5)
y = CModel.forward(x)
print(y)
print()

# class 에 직접 인자를 주면서 호출하면 내부의 __call__() 호출
# nn.Module 의 경우는 forward()를 호출하도록 되어있음
# 따라서 아래의 코드는 CModel.forward(x) 와 동일
y2 = CModel(x)
print(y2)