import torch
import torch.nn as nn
import torch.optim as opt
import torch.nn.functional as F


torch.manual_seed(777)

x_train = torch.FloatTensor([
    [73, 80, 75],
    [93, 88, 93],
    [89, 91, 90],
    [96, 98, 100],
    [73, 66, 70],
])
y_train = torch.FloatTensor([
    [152],
    [185],
    [180],
    [196],
    [142],
])

# TODO 1: 입력 특성 3개, 출력 1개인 nn.Linear
# 다중 선형회귀 함수 nn.Module 상속받은후 커스터마이즈 
class MultivariateLinearRegressionModel(nn.Modele):
    def __init__(self):
        super().__init__()
        self.lineaer = nn.Linear(3,1)

    def forward(self, x) :
        return self.lineaer(x)

model = MultivariateLinearRegressionModel()

# TODO 2: 평균제곱오차(MSE) 손실 함수

# TODO 3: SGD optimizer를 만들고 learning rate를 1e-5로 설정하세요.
optimizer = opt.SGD(model.parameters , lr = 1e-5)

nb_epochs = 20
for epoch in range(nb_epochs+1):
    
    # H(x) 계산
    prediction = model(x_train)
    
    # cost 계산
    cost = F.mse_loss(prediction, y_train)
    
    # cost로 H(x) 개선
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()
    
    # 20번마다 로그 출력
    print('Epoch {:4d}/{} Cost: {:.6f}'.format(
        epoch, nb_epochs, cost.item()
    ))

# 실행: