from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


torch.manual_seed(777)

data_path = Path(__file__).with_name("data-01-test-score.csv")

# TODO 1: np.loadtxt()로 CSV 파일을 불러오세요.
# 구분자는 쉼표이고 dtype은 np.float32입니다.

# TODO 2: 마지막 열을 제외한 값을 x_data에 저장하세요.

# TODO 3: 마지막 열을 y_data에 저장하되 (N, 1) 형태를 유지하세요.

# TODO 4: NumPy 배열을 torch Tensor로 변환하세요.

# TODO 5: 데이터와 Tensor shape가 올바른지 출력해서 확인하세요.

# TODO 6: nn.Linear, MSELoss, SGD를 사용해 모델을 학습하세요.

# TODO 7: 학습한 모델에 새로운 성적 데이터를 넣어 예측하세요.


# 실행:
# cd /Users/iyongsu/연습공간/Lab_task/practice
# /opt/homebrew/anaconda3/bin/python pytorch/lab-04-2_file_input_linear_regression.py
