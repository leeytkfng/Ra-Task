from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


torch.manual_seed(1)

data_path = Path(__file__).with_name('data-04-zoo.csv')


# Low-level: Softmax와 Cross Entropy의 내부 계산 확인
z = torch.rand(3, 5, requires_grad=True)
# dim=1의 5개 클래스 점수를 확률로 바꾸며 각 행의 합은 1이 된다.
hypothesis = F.softmax(z, dim=1)
y = torch.randint(5, (3,)).long()
y_one_hot = torch.zeros_like(hypothesis)
# y의 shape을 (3,)에서 (3, 1)로 늘려 scatter_가 각 행의 정답 열에 1을 기록하게 한다.
y_one_hot.scatter_(1, y.unsqueeze(1), 1)

torch.log(F.softmax(z, dim=1))

F.log_softmax(z,dim=1)

(y_one_hot * -torch.log(F.softmax(z,dim=1))).sum(dim=1).mean()

F.nll_loss(F.log_softmax(z,dim=1), y.long())

# cross_entropy는 logits에 log_softmax와 NLLLoss를 한 번에 적용하므로 Softmax를 먼저 쓰지 않는다.
F.cross_entropy(z,y)
print(F.cross_entropy(z,y))

xy = np.loadtxt(data_path, delimiter=',', dtype=np.float32)

x_train = torch.FloatTensor(xy[:, 0:-1])
y_train = torch.LongTensor(xy[:, [-1]]).squeeze(1)

print(x_train.shape)
print(len(x_train))
print(x_train[:5])

print(y_train.shape)
print(len(y_train))
print(y_train[:5])

nb_classes = 7
y_one_hot = torch.zeros((len(y_train)), nb_classes)
y_one_hot = y_one_hot.scatter(1, y_train.unsqueeze(1),1)

# Low-level: W와 b를 직접 선언한 학습
W = torch.zeros((16,7), requires_grad = True)
b = torch.zeros(7, requires_grad = True)
# optimizer 설정
optimizer = optim.SGD([W,b], lr=0.1)

nb_epochs = 1000
for epoch in range(nb_epochs + 1) :

    # Cost 계산 (2)
    z = x_train.matmul(W) + b # or .mm or @
    cost = F.cross_entropy(z, y_train)

    # cost로 H(x) 개선
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    # 100번마다 로그 출력
    if epoch % 100 == 0:
        print('Epoch {:4d}/{} Cost: {:.6f}'.format(
            epoch, nb_epochs, cost.item()
        ))


# High-level: nn.Module과 nn.Linear를 사용한 학습
class SoftmaxClassifierModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(16, 7)

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
# /opt/homebrew/anaconda3/bin/python pytorch/lab-06-2_fancy_softmax_classification.py
