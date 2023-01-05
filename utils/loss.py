import numpy
import torch
from torch import nn

pops = numpy.array([440.,  50., 939.,  60., 112., 625., 202.,  74., 998.,  57.,  43.,305.,  44.,  59., 548., 226.,  60.,  46.])
weights = torch.tensor(1/pops, dtype=torch.float32)
criterion = nn.CrossEntropyLoss(weight=weights)