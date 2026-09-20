from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


torch.manual_seed(1)

data_path = Path(__file__).with_name('data-04-zoo.csv')

z = torch.rand(3,5 , required_grad=True)
hypothesis = F.softmax(z,dim=1)
y = torch.randint(5,(3,)).long()
y_one_hot = torch.zeros_like(hypothesis)
y_one_hot.scatter_(1, y.unsqueeze(1),1)

# Low Level
torch.log(F.softmax(z, dim=1))

# High level
F.log_softmax(z,dim=1)

# Low level
(y_one_hot * -torch.log(F.softmax(z,dim=1))).sum(dim=1).mean()

# High level 
F.nll_loss(F.log_softmax(z,dim=1), y.long())

F.cross_entropy(z,y)



# 실행:
# cd /Users/iyongsu/연습공간/Lab_task/practice
# /opt/homebrew/anaconda3/bin/python pytorch/lab-06-2_fancy_softmax_classification.py
