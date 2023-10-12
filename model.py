
import torch.nn as nn
import torch.nn.functional as F

class SpaceInvadersModel(nn.Module):
  def __init__(self, n_actions=6):
    super(SpaceInvadersModel, self).__init__()

    nfs = (1, 4, 8, 16, 32, 64, 128, 256, 512) # 210x160 -> 105x80 -> 53x40 -> 27x20 -> 14x10 -> 7x5 -> 4x3 -> 2x2 -> 1x1
    hidden_layers = 2
    norm = nn.BatchNorm2d
    activate = nn.SiLU
    stride = 2
    kernel_size = 3
    padding = kernel_size // 2

    self.layers = nn.Sequential()
    for i in range(len(nfs) - 1):
        self.layers.append(nn.Conv2d(nfs[i], nfs[i + 1], kernel_size=kernel_size, stride=stride, padding=padding, bias=True))
        if i < len(nfs) - 2: # last layer isn't 2D
            self.layers.append(norm(nfs[i + 1]))
            self.layers.append(activate())
    self.layers.append(nn.Flatten())
    # add history of N moves
    for i in range(hidden_layers):
        self.layers.append(nn.Linear(nfs[-1], nfs[-1]))
        self.layers.append(activate())
    self.layers.append(nn.Linear(nfs[-1], n_actions))
    self.layers.append(nn.ReLU())
    self.layers.append(nn.Softmax(dim=1))

  def forward(self, x):
    return self.layers(x)
  