from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as opt

torch.manual_seed(1)
# TODO 1: np.loadtxt()로 CSV 파일을 불러오세요.
# 구분자는 쉼표이고 dtype은 np.float32입니다.

data_path = Path(__file__).with_name('data-01-test-score.csv')
xy = np.loadtxt(data_path, delimiter=',' , dtype=np.float32)

x_data = xy[:, 0:-1]
y_data = xy[:, [-1]]

print(x_data.shape)
print(len(x_data))
print(x_data[:5])

print(y_data.shape) # y_data shape
print(len(y_data))  # y_data 길이
print(y_data[:5])   # 첫 다섯 개

# TODO 2: 마지막 열을 제외한 값을 x_data에 저장하세요.
# 데이터
x_train = torch.FloatTensor(x_data)
y_train = torch.FloatTensor(y_data)

# 모델 초기화 
W = torch.zeros((3, 1), requires_grad = True)
b = torch.zeros(1, requires_grad = True)
# optimizer 
optimizer = opt.SGD([W, b], lr=1e-5)
nb_epochs = 20
for epoch in range(nb_epochs + 1):
    
    # H(x) 계산
    hypothesis = x_train.matmul(W) + b # or .mm or @

    # cost 계산
    cost = torch.mean((hypothesis - y_train) ** 2)

    # cost로 H(x) 개선
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    # 100번마다 로그 출력
    print('Epoch {:4d}/{} Cost: {:.6f}'.format(
        epoch, nb_epochs, cost.item()
    ))



# TODO 3: 마지막 열을 y_data에 저장하되 (N, 1) 형태를 유지하세요.

# TODO 4: NumPy 배열을 torch Tensor로 변환하세요.

# TODO 5: 데이터와 Tensor shape가 올바른지 출력해서 확인하세요.

# TODO 6: nn.Linear, MSELoss, SGD를 사용해 모델을 학습하세요.

# TODO 7: 학습한 모델에 새로운 성적 데이터를 넣어 예측하세요.


# 실행:
#  cd /Users/iyongsu/연습공간/Lab_task/practice
#  /opt/homebrew/anaconda3/bin/python pytorch/lab-04-2_file_input_linear_regression.py
