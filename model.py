
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, ni, nf, kernel_size=3, stride=1, padding=1, bias=True, norm=nn.BatchNorm2d, activate=nn.SiLU):
        super(ResidualBlock, self).__init__()
        self.layers = nn.Sequential()
        self.layers.append(nn.Conv2d(ni, nf, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias))
        if norm is not None:
            self.layers.append(norm(nf))
        if activate is not None:
            self.layers.append(activate())
        self.identity = nn.Conv2d(ni, nf, kernel_size=1, stride=stride, padding=0, bias=bias)

    def forward(self, x):
        return self.layers(x) + self.identity(x)


class SpaceInvadersModel(nn.Module):
  def __init__(self, n_actions=6, frames=3):
    super(SpaceInvadersModel, self).__init__()

    nfs = (frames, 8, 16, 32, 64, 128, 256, 512, 1024) # 210x160 -> 105x80 -> 53x40 -> 27x20 -> 14x10 -> 7x5 -> 4x3 -> 2x2 -> 1x1
    hidden_layers = 2
    normalize = nn.BatchNorm2d
    activate = nn.SiLU
    stride = 2
    kernel_size = 3
    padding = kernel_size // 2

    self.layers = nn.Sequential()
    for i in range(len(nfs) - 1):
        norm = normalize if i < len(nfs) - 2 else None
        act = activate if i < len(nfs) - 2 else None
        self.layers.append(ResidualBlock(nfs[i], nfs[i + 1], kernel_size=kernel_size, stride=stride, padding=padding, bias=True, norm=norm, activate=act))
    self.layers.append(nn.Flatten())
    # add history of N moves
    for i in range(hidden_layers):
        self.layers.append(nn.LayerNorm(nfs[-1]))
        self.layers.append(activate())
        self.layers.append(nn.Linear(nfs[-1], nfs[-1]))
    self.layers.append(nn.LayerNorm(nfs[-1]))
    self.layers.append(nn.Linear(nfs[-1], n_actions))
    self.layers.append(nn.LayerNorm(n_actions))
    self.layers.append(nn.Softmax(dim=1))

  def forward(self, x):
    #print(x.shape)
    for layer in self.layers:
        #print(layer, x.shape)
        x = layer(x)
        #print(x)
    #print(x[-1])
    return x
  