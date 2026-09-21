import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


# For reproductivity
torch.manual_seed(1)

x_train = torch.FloatTensor([
    [1, 2, 1],
    [1, 3, 2],
    [1, 3, 4],
    [1, 5, 5],
    [1, 7, 5],
    [1, 2, 5],
    [1, 6, 6],
    [1, 7, 7],
])
y_train = torch.LongTensor([2, 2, 2, 1, 1, 1, 0, 0])

x_test = torch.FloatTensor([
    [2, 1, 1],
    [3, 1, 2],
    [3, 3, 4],
])
y_test = torch.LongTensor([2, 2, 2])

# Model
class SoftmaxClassifierModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3, 3)
    def forward(self, x):
        return self.linear(x)

model = SoftmaxClassifierModel()    

# optimizer 
optimizer = optim.SGD(model.parameters(), lr=0.1)

def train(model, optimizer, x_train, y_train):
    model.train()
    nb_epochs = 20
    for epoch in range(nb_epochs + 1):

        # H(x) 게산
        prediction = model(x_train)

        # cost 계산 
        cost = F.cross_entropy(prediction, y_train)

        # cost로 H(x) 개선
        optimizer.zero_grad()
        cost.backward()
        optimizer.step()

        print('Epoch {:4d}/{} Cost: {:6f}'.format(
            epoch, nb_epochs, cost.item()
        ))

def test(model, x_test, y_test):
    model.eval()
    with torch.no_grad():
        prediction = model(x_test)
        predicted_classes = prediction.max(1)[1]
        correct_count = (predicted_classes == y_test).sum().item()
        cost = F.cross_entropy(prediction, y_test)

    print('Accuracy : {}% Cost: {:.6}'.format(
        correct_count / len(y_test) * 100 , cost.item()
    ))

train(model, optimizer , x_train, y_train)
test(model, x_test, y_test)

# lr 가 너무 크면 cost가 점점 늘어난다.
model = SoftmaxClassifierModel()
optimizer = optim.SGD(model.parameters(), lr = 1e5)
train(model, optimizer , x_train , y_train)

# 적절한 숫자로 시작해 발산하면 작게, cost가 줄어들지 않으면 크게 조정하자.
model = SoftmaxClassifierModel()
optimizer = optim.SGD(model.parameters(), lr=1e-10)
train(model, optimizer, x_train, y_train)

model = SoftmaxClassifierModel()
optimizer = optim.SGD(model.parameters(), lr=1e-1)
train(model, optimizer, x_train, y_train)

# 데이터 전처리 
# 데이터를 zero-center하고 nomalize를 진행한다.
x_train = torch.FloatTensor([[73, 80, 75],
                             [93, 88, 93],
                             [89, 91, 90],
                             [96, 98, 100],
                             [73, 66, 70]])
y_train = torch.FloatTensor([[152], [185], [180], [196], [142]])

mu = x_train.mean(dim=0)
sigma = x_train.std(dim=0)
norm_x_train = (x_train - mu) /sigma
print(norm_x_train)

class MultivariateLinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3, 1)
    def forward(self,x):
        return self.linear(x)

model = MultivariateLinearRegressionModel()
optimizer = optim.SGD(model.parameters(), lr=1e-1)

def train(model, optimizer , x_train, y_train):
    nb_epochs = 20 
    for epoch in range(nb_epochs):

        # H(x) 계산 
        prediction = model(x_train)

        # cost 계산 
        cost = F.mse_loss(prediction, y_train)

        # cost로 H(x) 개선
        optimizer.zero_grad()
        cost.backward()
        optimizer.step()

        print('Epoch {:4d}/{} Cost: {:.6f}'.format(
            epoch, nb_epochs, cost.item()
        ))

train(model, optimizer, norm_x_train, y_train)        

# Overfitting 
# 어떤 한 영역의 데이터에만 과도하게 맞춰저 있어 테스트 데이터에 좋은 성능을 내지 못할 수 있가.
# 더 많은 데이터 , 더 적은 양의 feature , Regularization
def train_with_regularization(model, optimizer ,x_train , y_train):
    nb_epochs = 20 
    for epoch in range(nb_epochs): 

        # H (x)게산
        prediction = model(x_train)

        # cost 계산 / mse 
        cost = F.mse_loss(prediction, y_train)

        # l2 norm 계산 
        l2_reg = 0 
        for param in model.parameters(): 
            l2_reg += torch.norm(param)

        cost += l2_reg

        # cost로 H(x) 개선
        optimizer.zero_grad()
        cost.backward()
        optimizer.step()

        print('Epoch {:4d}/{} Cost: {:.6f}'.format(
            epoch+1, nb_epochs, cost.item()
        ))

model = MultivariateLinearRegressionModel()
optimizer = optim.SGD(model.parameters(), lr=1e-1)
train_with_regularization(model, optimizer, norm_x_train, y_train)




# 실행:
# cd /Users/iyongsu/연습공간/Lab_task/practice
# /opt/homebrew/anaconda3/bin/python pytorch/lab-07-1_tips.py
