import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as opt

torch.manual_seed(777)

# FloatTensor로 변경하여 float32 연산 수행
x_train = torch.FloatTensor([[1], [2], [3]])
y_train = torch.FloatTensor([[1], [2], [3]])
# requires_grad=True로 설정하면 학습을 통해 값이 변경됨
W = torch.zeros(1, requires_grad=True)
b = torch.zeros(1, requires_grad=True)
# optimizer 수식 설정
optimizer = opt.SGD([W, b], lr=0.001)
# epoch = 1000
nb_epochs = 1000
# 학습 진행
for epoch in range(nb_epochs + 1):
    # H(x) 게산 식 선언
    hypothesis = x_train * W + b

    # Loss function 계산
    cost = torch.mean((hypothesis - y_train) ** 2)

    # optimizer 초기화, Backpropagation 수행, optimizer step 수행
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    # 100번마다 로그 출력
    if epoch % 100 == 0:
        print('Epoch {:4d}/{} W: {:.3f}, b: {:.3f} Cost: {:.6f}'.format(
            epoch, nb_epochs, W.item(), b.item(), cost.item()
        ))
