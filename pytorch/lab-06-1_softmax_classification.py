import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn


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

# Softmax는 각 행의 logits를 클래스 확률로 변환하며 확률의 합은 1이다.
z = torch.rand(3, 5 , requires_grad = True)
hypothesis = F.softmax(z, dim= 1)
print(hypothesis)

y = torch.randint(5, (3,)).long()
print(y)

y_one_hot = torch.zeros_like(hypothesis)
y_one_hot.scatter_(1, y.unsqueeze(1), 1)

cost = (y_one_hot * -torch.log(hypothesis)).sum(dim=1).mean()
print(cost)

# log_softmax는 log(softmax)와 같은 계산을 더 안정적으로 수행한다.
torch.log(F.softmax(z, dim=1))

F.log_softmax(z, dim=1)

# Cross Entropy = LogSoftmax + NLLLoss
(y_one_hot * -torch.log(F.softmax(z, dim=1))).sum(dim=1).mean()

F.nll_loss(F.log_softmax(z, dim=1), y)

F.cross_entropy(z,y)

# W와 b를 직접 학습하는 Softmax 분류
x_train = [[1, 2, 1, 1],
           [2, 1, 3, 2],
           [3, 1, 3, 4],
           [4, 1, 5, 5],
           [1, 7, 5, 5],
           [1, 2, 5, 6],
           [1, 6, 6, 6],
           [1, 7, 7, 7]]


y_train = [2, 2, 2, 1, 1, 1, 0, 0]


x_train = torch.FloatTensor(x_train)
y_train = torch.LongTensor(y_train)

W = torch.zeros((4, 3), requires_grad=True)
b = torch.zeros(1, requires_grad=True)
optimizer = optim.SGD([W, b], lr=0.1)

nb_epochs = 1000
for epoch in range(nb_epochs + 1):

    # cross_entropy에는 Softmax를 적용하지 않은 logits를 전달한다.
    z = x_train.matmul(W) + b
    cost = F.cross_entropy(z, y_train)

    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print('Epoch {:4d}/{} Cost: {:.6f}'.format(
            epoch, nb_epochs, cost.item()
        ))

# nn.Linear를 사용하는 고수준 모델
class SoftmaxClassifierModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 3)

    def forward(self, x):
        return self.linear(x)

model = SoftmaxClassifierModel()

optimizer = optim.SGD(model.parameters(), lr=0.1)

nb_epochs = 1000
for epoch in range(nb_epochs + 1):

    prediction = model(x_train)

    cost = F.cross_entropy(prediction, y_train)

    optimizer.zero_grad()
    cost.backward()
    optimizer.step()
    
    if epoch % 100 == 0:
        print('Epoch {:4d}/{} Cost: {:.6f}'.format(
            epoch, nb_epochs, cost.item()
        ))


# 실행:
# cd /Users/iyongsu/연습공간/Lab_task/practice
# /opt/homebrew/anaconda3/bin/python pytorch/lab-06-1_softmax_classification.py
