import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from pathlib import Path

torch.manual_seed(777)

data_path = Path(__file__).with_name('data-03-diabetes.csv')
xy = np.loadtxt(data_path, delimiter=',', dtype=np.float32)
x_data = xy[:, 0:-1]
y_data = xy[:, [-1]]
x_train = torch.FloatTensor(x_data)
y_train = torch.FloatTensor(y_data)


class BinaryClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(8, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.linear(x))



# TODO 1: 입력 특성 2개를 하나의 값으로 변환하는 nn.Linear 모델을 만드세요.
# training 데이터 설정 
# x_data = [[1, 2], [2, 3], [3, 1], [4, 3], [5, 3], [6, 2]]
# y_data = [[0], [0], [0], [1], [1], [1]]

# x_train = torch.FloatTensor(x_data)
# y_train = torch.FloatTensor(y_data)

# TODO 2: Linear 모델의 출력에 sigmoid를 적용해 확률을 구하세요.
print('e^1 equals: ', torch.exp(torch.FloatTensor([1])))

W = torch.zeros((x_train.shape[1], 1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)

hypothesis = 1/ (1 + torch.exp(-(x_train.matmul(W) + b )))

model = BinaryClassifier()

# TODO 3: Binary Cross Entropy를 직접 계산하거나 nn.BCELoss를 사용하세요.

hypothesis = torch.sigmoid(x_train.matmul(W) + b)

losses = -(y_train * torch.log(hypothesis) + 
           (1 - y_train) * torch.log(1 - hypothesis))

cost = losses.mean()

# Computing with F 
F.binary_cross_entropy(hypothesis,y_train)

# x_data = [[1, 2], [2, 3], [3, 1], [4, 3], [5, 3], [6, 2]]
# y_data = [[0], [0], [0], [1], [1], [1]]
# x_train = torch.FloatTensor(x_data)
# y_train = torch.FloatTensor(y_data)

W = torch.zeros((x_train.shape[1], 1), requires_grad=True)
b = torch.zeros(1, requires_grad = True)

# optimizer 
optimizer = optim.SGD([W,b] , lr = 1)

nb_epochs = 100
for epoch in range(nb_epochs + 1) : 

    # Cost 계산 
    hypothesis = torch.sigmoid(x_train.matmul(W) + b)
    # Cross Entropy 직접 선언 버전
    # cost = -(y_train * torch.log(hypothesis) + 
    #         (1 - y_train) * torch.log( 1 - hypothesis)).mean()
    cost = F.binary_cross_entropy(hypothesis,y_train)

    #cost로 H(X) 개선
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    # 100번마다 로그를 출력한다.
    if epoch % 10 == 0:
        prediction = hypothesis >= torch.FloatTensor([0.5])
        correct_prediction = prediction.float() == y_train
        accuracy = correct_prediction.sum().item() / len(correct_prediction)
        print('Epoch {:4d}/{} Cost: {:.6f} Accuracy {:2.2f}% '.format(
            epoch, nb_epochs , cost.item(), accuracy * 100,
        ))
