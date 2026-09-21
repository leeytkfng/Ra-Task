import torch
import torch.nn as nn
import torch.optim as optim


torch.manual_seed(777)

num_classes = 10
batch_size = 100
num_epochs = 15

# TODO 1: torchvision의 MNIST train/test dataset을 준비하세요.
# 이미지를 Tensor로 변환하는 transform을 설정하세요.

# TODO 2: train/test DataLoader를 만드세요.
# train loader만 shuffle=True로 설정하세요.

# TODO 3: 28x28 이미지를 784차원으로 펼쳐 10개 logits를 출력하는 모델을 만드세요.

# TODO 4: CrossEntropyLoss와 optimizer를 준비하세요.

# TODO 5: 미니배치 학습 루프를 작성하고 epoch별 평균 loss를 출력하세요.

# TODO 6: model.eval()과 torch.no_grad()를 사용해 test accuracy를 계산하세요.

# TODO 7: 테스트 이미지 하나의 실제 label과 예측 결과를 확인하세요.


# 실행:
# cd /Users/iyongsu/연습공간/Lab_task/practice
# /opt/homebrew/anaconda3/bin/python pytorch/lab-07-2_mnist.py
