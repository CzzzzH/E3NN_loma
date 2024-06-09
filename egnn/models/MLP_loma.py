from torch import nn
from ops.torch_ops import LomaLinear, LomaReLU

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
