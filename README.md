# Deep Learning Study — RA Task

모두를 위한 딥러닝의 TensorFlow 실습을 읽고, 같은 학습 원리를 PyTorch로 직접
구현·학습·변형하는 공부용 저장소입니다.

완성된 코드를 모으는 것보다 다음 과정을 스스로 설명할 수 있는 것을 목표로 합니다.

```text
모델과 수식 이해
→ TensorFlow 원본 코드 분석
→ PyTorch 직접 구현
→ 학습 및 결과 재현
→ 한 변수만 변경하는 실험
→ 정량·정성 결과 분석
→ 보고서 작성
```

## 과제 구현 목표

### 1. 딥러닝 기본기

- Tensor와 shape 이해
- hypothesis, loss, gradient의 관계 설명
- `backward()`를 통한 gradient 계산 이해
- `optimizer.zero_grad() → backward() → optimizer.step()` 학습 과정 구현
- train/test 분리와 일반화 성능 확인
- activation, initialization, dropout의 효과 비교

### 2. Linear Regression

현재 진행 중인 첫 번째 과제입니다.

1. $\hat{y}=Wx+b$를 PyTorch Tensor로 구현
2. Mean Squared Error 계산
3. `backward()`와 SGD로 $W$, $b$ 학습
4. 학습 후 $W\rightarrow1$, $b\rightarrow0$ 수렴 확인
5. 여러 $W$에 대한 Cost Function 계산 및 시각화
6. optimizer 없이 Gradient Descent update 식 직접 구현
7. 수동 gradient와 PyTorch autograd gradient 비교
8. learning rate `0.1`, `0.01`, `0.001`의 수렴 속도 비교

Linear Regression 보고서는 다음 흐름으로 정리합니다.

```text
Hypothesis와 Cost Function
→ PyTorch 구현
→ Cost Function 시각화
→ Gradient Descent 직접 구현
→ PyTorch SGD와 비교
→ Learning Rate 실험
→ 결과 분석
```

### 3. 이후 학습 범위

- Logistic Regression
- Softmax Classification
- Train/Test와 Normalization
- XOR와 Backpropagation
- ReLU, Initialization, Dropout
- CNN 구현 및 비교 실험
- RNN 개념 확인

CNN부터는 모델 구조, Tensor shape, train/test loss와 accuracy, 변경 실험 결과를
더 자세히 기록합니다.

## 디렉터리

```text
.
├── tensorflow_original/
│   └── TensorFlow 원본의 작업용 복사본
└── pytorch/
    └── 직접 작성하는 PyTorch 구현
```

- `tensorflow_original/`: 원본의 수식과 데이터 흐름을 분석하고 주석을 작성합니다.
- `pytorch/`: 원본을 이해한 뒤 같은 Lab 번호로 직접 구현합니다.

## 현재 파일

| 파일 | 내용 | 상태 |
|---|---|---|
| `tensorflow_original/lab-02-1-linear_regression.py` | TensorFlow Linear Regression 원본 | 참고용 |
| `pytorch/lab-02-1-linear_regression.py` | PyTorch Linear Regression | 구현 및 학습 가능 |
| `pytorch/lab-03_minimizing_cost.py` | Cost/Gradient/SGD/LR 비교 | 직접 구현할 TODO |
| `pytorch/lab-04_multivariable_linear_regression.py` | 행렬곱 기반 다변수 선형회귀 | 직접 구현 중 |
| `pytorch/lab-04-1_multivariable_linear_regression_high_level.py` | `nn.Linear` 기반 다변수 선형회귀 | 직접 구현할 TODO |
| `pytorch/lab-04-2_file_input_linear_regression.py` | CSV 파일 입력 기반 다변수 선형회귀 | 직접 구현할 TODO |

## 실행

저장소 루트에서 실행합니다.

```bash
python pytorch/lab-02-1-linear_regression.py
```

현재 확인한 학습 환경에서는 다음 인터프리터를 사용합니다.

```bash
/opt/homebrew/anaconda3/bin/python pytorch/lab-02-1-linear_regression.py
```

## 실험 원칙

- 원본과 직접 구현한 코드를 구분합니다.
- 한 번에 전체 Lab을 완성하지 않고 순서대로 진행합니다.
- 비교 실험에서는 한 변수만 변경합니다.
- learning rate를 비교할 때 초기값, 데이터, epoch, optimizer를 동일하게 유지합니다.
- 실행하지 않은 결과나 설명하지 못하는 결과는 보고서에 사용하지 않습니다.
- 최종 정확도뿐 아니라 loss curve, Tensor shape, 파라미터 수와 실패 결과도 기록합니다.
- 새 실습 파일 맨 아래에는 저장소 루트로 이동하는 `cd` 명령과 실행 명령을 주석으로 적습니다.

## 출처

- [hunkim/DeepLearningZeroToAll](https://github.com/hunkim/DeepLearningZeroToAll)
- 모두를 위한 딥러닝 시즌 1·2
