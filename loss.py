from torch import nn
from torch.nn import functional as F

def Loss(y_hat, y):
    criterion = nn.NLLLoss()
    return criterion(F.log_softmax(y_hat, dim=1), y.long())
