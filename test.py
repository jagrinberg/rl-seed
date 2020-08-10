import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

mean1 = torch.DoubleTensor([[1,2],[3,4]])
mean2 = torch.DoubleTensor([[2,2],[2,4]])
stddev1 = torch.DoubleTensor([[5,4],[3,2]])
stddev2 = torch.DoubleTensor([[1,2],[1,2]])

ans = torch.log(stddev2/stddev1)-.5+(torch.pow(stddev1, 2)+torch.pow(mean1-mean2, 2))/(2*torch.pow(stddev2, 2))


print(ans)
