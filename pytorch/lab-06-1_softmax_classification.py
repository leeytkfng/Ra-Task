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

# Low level
torch.log(F.softmax(z, dim=1))

# High level
F.log_softmax(z, dim=1)

# Low level
(y_one_hot * -torch.log(F.softmax(z, dim=1))).sum(dim=1).mean()

# High level
F.nll_loss(F.log_softmax(z, dim=1), y)

# TODO 1: 입력 특성 4개와 클래스 3개에 맞는 W와 b를 만드세요.

# TODO 2: logits = XW+b를 계산하고 dim=1 방향으로 softmax를 적용하세요.

# TODO 3: one-hot label을 이용해 Cross Entropy를 직접 계산하세요.

# TODO 4: SGD optimizer를 만들고 learning rate를 0.1로 설정하세요.

# TODO 5: 2001번 학습하는 반복문을 작성하세요.

# TODO 6: 새로운 입력의 클래스별 확률과 argmax 예측 클래스를 출력하세요.


# 실행:
# cd /Users/iyongsu/연습공간/Lab_task/practice
# /opt/homebrew/anaconda3/bin/python pytorch/lab-06-1_softmax_classification.py
