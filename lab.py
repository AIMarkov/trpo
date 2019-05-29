import torch.nn as nn
import torch
# linear=nn.Linear(2,3)
# linear.weight.data.fill_(1.0)
# print(linear.weight.data)
# linear.weight.data.mul_(0.0)
# print(linear.weight.data)
# action_log_std = nn.Parameter(torch.zeros(1, 3))
# #print(action_log_std)
# b=action_log_std.expand([4,3])
# print(b)
# import numpy as np
# a=np.asarray([12,3,4,5,6132])
# a[...]=[13,3,4,5,6132]
# print(a[...])
# assert 1 == 2
b=torch.FloatTensor([1.3232424254,2,3,4])
print(b[0,0])