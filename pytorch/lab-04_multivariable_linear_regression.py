import torch 
import torch.nn as nn 
import torch.nn.functional as F
import torch.optim as opt


torch.manual_seed(1)

# 데이터 FloatTensor
# x1_train = torch.FloatTensor([[73], [93], [89], [96], [73]])
# x2_train = torch.FloatTensor([[80], [88], [91], [98], [66]])
# x3_train = torch.FloatTensor([[75], [93], [90], [100], [70]])
# y_train = torch.FloatTensor([[152], [185], [180], [196], [142]])

x_train = torch.FloatTensor([[73, 80, 75],
                             [93, 88, 93],
                             [89, 91, 90],
                             [96, 98, 100],
                             [73, 66, 70]])
y_train = torch.FloatTensor([[152], [185], [180], [196], [142]])

print(x_train.shape)
print(y_train.shape) 


# Model initalization 
# w1 = torch.zeros(1, requires_grad=True)
# w2 = torch.zeros(1, requires_grad=True)
# w3 = torch.zeros(1, requires_grad=True)
# b = torch.zeros(1, requires_grad=True)

# 모델 초기화 
W = torch.zeros((3,1), requires_grad = True)
b = torch.zeros(1, requires_grad = True)




# Optimizer 
# 관례적으로 1e-5를 설정함으로써 학습을 안정적으로 설정한다.
# optimizer = opt.SGD([w1, w2, w3, b], lr=1e-5)
 
optimizer = opt.SGD([W,b] , lr = 1e-5)

nb_epochs = 20

for epoch in range(nb_epochs + 1):

    # H(x)
    # hypothesis = x1_train * w1 + x2_train * w2 + x3_train * w3 + b
    hypothesis = x_train.matmul(W) + b # or . mm or @ 


    # Loss function 계산 
    loss = torch.mean((hypothesis - y_train) ** 2)

    # cost로 H(x) 개선
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # 100번 마다 로그를 출력한다 .
    # if epoch % 100 == 0:
    #     print('Epoch {:4d}/{} w1: {:.3f} w2: {:.3f} w3: {:.3f} b: {:.3f} Cost: {:.6f}'.format(
    #         epoch, nb_epochs, w1.item(), w3.item(), w3.item(), b.item(), loss.item()
    #     ))
    
    print('Epoch {:4d}/{} hypothesis: {} Cost: {:.6f}'.format(
        epoch, nb_epochs, hypothesis.squeeze().detach(), loss.item()
    ))


# 실행:
# cd /Users/iyongsu/연습공간/Lab_task/practice
# /opt/homebrew/anaconda3/bin/python pytorch/lab-04_multivariable_linear_regression.py
