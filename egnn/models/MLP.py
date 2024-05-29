from ops.torch_ops import LomaLinear, LomaReLU
from torch import nn

class MLP(nn.Module):

    def __init__(self):
        super(MLP, self).__init__()
        self.flatten = nn.Flatten()
        self.layers = nn.Sequential(
            LomaLinear(28 * 28, 512),
            LomaReLU(),
            LomaLinear(512, 512),
            LomaReLU(),
            LomaLinear(512, 1)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.layers(x)
        return logits

class MLPRef(nn.Module):

    def __init__(self):
        super(MLPRef, self).__init__()
        self.flatten = nn.Flatten()
        self.layers = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.layers(x)
        return logits