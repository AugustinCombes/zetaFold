import numpy as np
import torch
from torch import nn

pops = np.array([440.,  50., 939.,  60., 112., 625., 202.,  74., 998.,  57.,  43.,305.,  44.,  59., 548., 226.,  60.,  46.])
weights = 1/pops
weights = weights/np.mean(weights)
weights = torch.tensor(weights, dtype=torch.float32)
criterion = nn.CrossEntropyLoss(weight=weights)