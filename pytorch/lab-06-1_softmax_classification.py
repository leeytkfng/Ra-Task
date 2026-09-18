import torch
import torch.nn.functional as F
import torch.optim as optim


torch.manual_seed(1)

x_train = torch.FloatTensor([
    [1, 2, 1, 1],
    [2, 1, 3, 2],
    [3, 1, 3, 4],
    [4, 1, 5, 5],
    [1, 7, 5, 5],
    [1, 2, 5, 6],
    [1, 6, 6, 6],
    [1, 7, 7, 7],
])
y_train = torch.FloatTensor([
    [0, 0, 1],
    [0, 0, 1],
    [0, 0, 1],
    [0, 1, 0],
    [0, 1, 0],
    [0, 1, 0],
    [1, 0, 0],
    [1, 0, 0],
])

print('x_train shape:', x_train.shape)
print('y_train shape:', y_train.shape)

z = torch.FloatTensor([1,2,3])

hypothesis = F.softmax(z, dim = 0)
print(hypothesis)

hypothesis.sum()

# 
z = torch.rand(3, 5 , requires_grad = True)
hypothesis = F.softmax(z, dim= 1)
print(hypothesis)

y = torch.randint(5, (3,)).long()
print(y)

y_one_hot = torch.zeros_like(hypothesis)
y_one_hot.scatter_(1, y.unsqueeze(1), 1)

cost = (y_one_hot * -torch.log(hypothesis)).sum(dim=1).mean()
print(cost)

# 저수준 
torch.log(F.softmax(z, dim=1))

# 고수준
F.log_softmax(z, dim=1)

# 저수준 구현
(y_one_hot * -torch.log(F.softmax(z, dim=1))).sum(dim=1).mean()

# 고수준 구현
F.nll_loss(F.log_softmax(z, dim=1), y)

F.cross_entropy(z,y)
