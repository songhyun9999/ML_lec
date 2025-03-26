import torch
import torch.nn.functional as F

torch.manual_seed(777)

wsum = torch.randn(3,5,requires_grad=True)
print(wsum)
hypothesis = F.softmax(wsum,dim=1)
print()
print(hypothesis)
print()

y = torch.randint(5,(3,)).long()
print(y)
print()

y_one_hot = torch.zeros_like(hypothesis)
print(y_one_hot)
print()

y_one_hot = y_one_hot.scatter(1,y.unsqueeze(dim=1),1)
print(y_one_hot)
print()

print(-(y_one_hot * torch.log(F.softmax(wsum,dim=1))).sum(dim=1))
print(-(y_one_hot * torch.log(F.softmax(wsum,dim=1))).sum(dim=1).mean())
print(-(y_one_hot * torch.log_softmax(wsum,dim=1)).sum(dim=1).mean())
print(F.cross_entropy(wsum,y))