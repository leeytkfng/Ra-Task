# 직접 구현 작업 규칙

이 폴더는 사용자가 직접 생각하고 수정하는 공간입니다. 완성된 정답 파일을 먼저
복사하지 않습니다.

## Lab 하나를 공부하는 순서

### 1. 강의와 수식

아래 네 항목을 코드보다 먼저 적습니다.

- 문제는 regression인가 classification인가?
- 입력 `X`, 정답 `Y`, 파라미터 `W`, 출력의 shape는 무엇인가?
- hypothesis 식은 무엇인가?
- loss와 update 식은 무엇인가?

### 2. TensorFlow 원본 읽기

`tensorflow_original/`의 작업 파일에 다음 주석을 직접 답합니다.

```python
# STUDY: 이 Tensor의 shape는?
# STUDY: 이 연산이 구현하는 수식은?
# STUDY: 학습되는 파라미터는 무엇인가?
# STUDY: 이 줄은 학습/평가 중 언제 필요한가?
```

TensorFlow 문법을 외우는 것보다 데이터와 gradient 흐름을 설명하는 것이 목표입니다.

### 3. 직접 구현

원본을 닫고 `pytorch/`에 같은 학습 알고리즘을 작성합니다. 처음에는 다음 구조만
놓고 하나씩 채웁니다.

```text
data
→ model
→ forward
→ loss
→ zero_grad
→ backward
→ optimizer.step
→ evaluation
```

완성 참고 구현은 직접 시도하고 막힌 지점을 기록한 뒤에만 확인합니다.

### 4. Baseline 실행

아래 항목을 `../report/EXPERIMENT_LOG.md`에 바로 기록합니다.

- 날짜와 코드 파일
- seed, epoch, batch size, optimizer, learning rate
- train/validation/test loss와 accuracy
- 입력부터 출력까지 주요 shape
- 실행 중 관찰한 현상

### 5. Controlled experiment

한 실행에서는 변수 하나만 변경합니다.

```text
activation: ReLU → Sigmoid
learning rate: 0.001 → 0.01
dropout: 0.0 → 0.5
initialization: default → Xavier
capacity: channel 또는 hidden dimension 변경
```

### 6. 분석

“A가 더 높았다”에서 끝내지 않고 다음에 답합니다.

- 학습 곡선은 어떻게 달라졌는가?
- train-test gap은 어떻게 달라졌는가?
- 수업의 어떤 개념으로 차이를 설명할 수 있는가?
- 같은 결론을 내리려면 반복 실험이나 추가 통제가 필요한가?

## 파일명

원본과 대응 관계가 보이도록 번호를 유지합니다.

```text
tensorflow_original/lab-02-1-linear_regression.py
pytorch/lab-02-1-linear_regression.py
```

PyTorch 파일의 TODO는 직접 채웁니다. 명시적으로 참고 구현을 요청하기 전에는
완성 코드로 교체하지 않습니다.

# Ra-Task
