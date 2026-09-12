# Lab 3: Minimizing Cost
#
# Linear Regression의 Cost Function, Gradient Descent, PyTorch SGD가
# 서로 어떻게 연결되는지 확인하는 실습이다.
#
# 전체 흐름
# Cost Function 형태 확인
# -> Gradient Descent 직접 구현
# -> PyTorch SGD 결과와 비교
# -> Learning Rate 변경 실험

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as opt
import numpy as np
# 시각적 자료 출력을 위해 matplotlib.pyplot을 import한다.
import matplotlib.pyplot as plt


# -----------------------------------------------------------------------------
# Lab 3-1. Cost Function 시각화
# -----------------------------------------------------------------------------
# 목적
# - 학습을 수행하지 않고 W 값을 직접 바꾸면서 J(W)를 계산한다.
# - b는 0으로 고정하여 W와 cost의 관계만 확인한다.
# - y=x 데이터에서는 W=1일 때 cost가 최소가 되는지 그래프로 확인한다.
#
# 수식
# J(W) = (1/N) * sum((W*x_i - y_i)^2)
#
# TODO
# 1. torch와 matplotlib.pyplot을 import한다.
# 2. x_train=[1,2,3], y_train=[1,2,3]을 float Tensor로 만든다.
# 3. W 후보를 -3.0부터 5.0까지 일정한 간격으로 만든다.
# 4. 각 W 후보에 대해 hypothesis와 cost를 계산한다.
# 5. W 값과 cost를 각각 history에 저장한다.
# 6. x축은 W, y축은 Cost로 그래프를 그린다.
# 7. W=1 위치와 최소 cost를 그래프에서 표시한다.
#
# 기록
# - 최소 cost가 나온 W
# - 그래프가 포물선 형태가 되는 이유
# - backward()와 optimizer가 필요 없는 이유


# -----------------------------------------------------------------------------
# Lab 3-2. Gradient Descent 직접 구현
# -----------------------------------------------------------------------------
# 목적
# - optimizer와 backward() 없이 gradient와 update 식을 직접 작성한다.
# - 초기 W에서 시작해 W가 cost의 최솟값인 1로 이동하는지 확인한다.
# - 원본 Lab 3과 맞추기 위해 이 실험에서는 b를 사용하지 않는다.
#
# 수식
# hypothesis = W*x
# J(W) = (1/N) * sum((W*x_i - y_i)^2)
# dJ/dW = (2/N) * sum((W*x_i - y_i)*x_i)
# W <- W - learning_rate * dJ/dW
#
# 주의
# - TensorFlow 원본 lab-03-2의 gradient 식에는 상수 2가 생략되어 있다.
# - 상수 2를 포함한 식과 원본 식은 같은 방향이지만 이동 크기가 다르다.
#
# TODO
# 1. W를 정답과 떨어진 값으로 초기화한다.
# 2. 매 step마다 hypothesis와 cost를 계산한다.
# 3. 미분식으로 gradient를 직접 계산한다.
# 4. gradient 추적 없이 W를 직접 update한다.
# 5. step, W, gradient, cost를 history에 저장한다.
# 6. W와 cost가 step에 따라 변하는 모습을 시각화한다.
#
# 기록
# - 첫 step과 마지막 step의 W, gradient, cost
# - gradient 부호와 W의 이동 방향
# - gradient 식의 상수 2가 수렴 속도에 미치는 영향


# -----------------------------------------------------------------------------
# Lab 3-3. PyTorch SGD Optimizer와 비교
# -----------------------------------------------------------------------------
# 목적
# - 직접 구현한 update와 PyTorch optimizer가 같은 역할을 하는지 비교한다.
#
# TODO
# 1. 수동 구현과 optimizer 구현의 초기 W, 데이터, learning rate를 같게 한다.
# 2. 두 구현 모두 b를 사용하거나 모두 b를 제거한다.
# 3. 같은 step에서 W와 cost를 나란히 기록한다.
# 4. W.grad와 수식으로 계산한 gradient를 비교한다.
# 5. 두 gradient의 최대 절대 오차를 계산한다.
#
# 확인할 대응 관계
# TensorFlow: GradientDescentOptimizer(...).minimize(cost)
# PyTorch: optimizer.zero_grad() -> cost.backward() -> optimizer.step()


# -----------------------------------------------------------------------------
# Lab 3-4. Learning Rate에 따른 수렴 비교
# -----------------------------------------------------------------------------
# 비교 값
# learning_rate = 0.1, 0.01, 0.001
#
# 통제 조건
# - 같은 초기 W와 b
# - 같은 학습 데이터와 epoch 수
# - 같은 optimizer와 loss
# - learning rate만 변경
#
# TODO
# 1. learning rate마다 파라미터와 optimizer를 새로 초기화한다.
# 2. 각 실험의 cost history를 별도로 저장한다.
# 3. 한 그래프에 세 loss curve와 범례를 표시한다.
# 4. 필요하면 y축 log scale 그래프도 만든다.
# 5. 최종 W, b, cost와 수렴 step을 표로 정리한다.
#
# 분석 질문
# - learning rate가 너무 크면 loss와 파라미터가 어떻게 움직이는가?
# - learning rate가 너무 작으면 같은 epoch에서 왜 덜 수렴하는가?
# - 최종 cost만으로 수렴 속도를 판단할 수 있는가?
# - 가장 적절한 learning rate와 그 근거는 무엇인가?


# -----------------------------------------------------------------------------
# 보고서 구성
# -----------------------------------------------------------------------------
# 1. Hypothesis와 Cost Function
# 2. PyTorch 기반 Linear Regression 구현
# 3. Cost Function 시각화
# 4. Gradient Descent 직접 구현
# 5. PyTorch SGD 적용 및 수동 구현과 비교
# 6. Learning Rate에 따른 수렴 비교
# 7. 결과 분석과 배운 점
