from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


torch.manual_seed(777)

data_path = Path(__file__).with_name('data-04-zoo.csv')

# TODO 1: np.loadtxt()로 Zoo CSV 데이터를 불러오세요.

# TODO 2: 앞의 16개 열을 x_data, 마지막 열을 y_data로 분리하세요.

# TODO 3: NumPy 배열을 PyTorch Tensor로 변환하고 shape를 확인하세요.

# TODO 4: 입력 특성 16개와 클래스 7개에 맞는 nn.Linear 모델을 만드세요.

# TODO 5: nn.CrossEntropyLoss와 SGD optimizer를 만드세요.
# CrossEntropyLoss에는 softmax 확률이 아닌 logits를 전달합니다.
# 정답은 one-hot이 아닌 1차원 LongTensor 클래스 인덱스여야 합니다.

# TODO 6: 2001번 학습하며 loss와 accuracy를 계산하세요.

# TODO 7: 전체 데이터의 예측 클래스와 실제 클래스를 비교하세요.


# 실행:
# cd /Users/iyongsu/연습공간/Lab_task/practice
# /opt/homebrew/anaconda3/bin/python pytorch/lab-06-2_fancy_softmax_classification.py
