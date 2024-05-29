from ops.torch_ops import LomaLinear
from torch import nn

class MLP(nn.Module):

    def __init__(self):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            LomaLinear(10, 10),
            nn.ReLU(),
            LomaLinear(10, 10),
        )

    def forward(self, input):
        return self.layers(input)