import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

torch.manual_seed(1)


# =========================================================
# 1. Training Data
# =========================================================

x_train = torch.tensor([[1.0], [2.0], [3.0]])
y_train = torch.tensor([[1.0], [2.0], [3.0]])


# =========================================================
# 2. Data Visualization
# =========================================================

plt.scatter(x_train.numpy(), y_train.numpy(), label='Training Data')

xs = np.linspace(1, 3, 100)
plt.plot(xs, xs, label='y = x')

plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()


# =========================================================
# 3. Cost Function Visualization
# =========================================================

w_values = np.linspace(-5, 7, 1000)
cost_values = []

for w in w_values:
    hypothesis = x_train * w
    cost = torch.mean((hypothesis - y_train) ** 2)

    cost_values.append(cost.item())


plt.plot(w_values, cost_values)
plt.xlabel('W')
plt.ylabel('Cost')
plt.title('Cost Function')
plt.show()


# =========================================================
# 4. Gradient Descent 직접 계산
# =========================================================

W = 0.0
lr = 0.01

hypothesis = W * x_train

# Cost = mean((Wx - y)^2)
# dCost/dW = 2 * mean((Wx - y) * x)
gradient = 2 * torch.mean((hypothesis - y_train) * x_train)

print("Gradient:", gradient.item())

W -= lr * gradient.item()

print("Updated W:", W)

# 선형 회귀 함수 클래스 커스터마이즈 
# class LinearRegressionModel(nn.Modele):
    # def __init__(self):
    #     super().__init__()
    #     self.lineaer = nn.Linear(1,1)

    # def forward(self, x) :
    #     return self.lineaer(x)


# =========================================================
# 5. PyTorch SGD
# =========================================================

W = torch.zeros(1, requires_grad=True)

optimizer = optim.SGD([W], lr=0.15)

nb_epochs = 10

for epoch in range(nb_epochs + 1):

    # Forward
    hypothesis = x_train * W

    # Cost
    cost = torch.mean((hypothesis - y_train) ** 2)

    print(
        f'Epoch {epoch:2d}/{nb_epochs} '
        f'W: {W.item():.3f} '
        f'Cost: {cost.item():.6f}'
    )

    # Backward
    optimizer.zero_grad()
    cost.backward()

    # W <- W - lr * gradient
    optimizer.step()


# 실행:
# cd /Users/iyongsu/연습공간/Lab_task/practice
# /opt/homebrew/anaconda3/bin/python pytorch/lab-03_minimizing_cost.py
