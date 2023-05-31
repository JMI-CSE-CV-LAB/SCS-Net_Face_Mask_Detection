import torch
from torch import nn
from functools import partial
from model.scs_layer import SharpenedCosineSimilarity, AbsPool
from model.se_net import SENet

class model(nn.Module):
  def __init__(self):
        super().__init__()
        
        MaxAbsPool2d = partial(AbsPool, nn.MaxPool2d)
        self.pool = MaxAbsPool2d(kernel_size=2, stride=2, ceil_mode=True)

        self.scs1 = SharpenedCosineSimilarity(
            in_channels=3, out_channels=8, kernel_size=3, padding=0)
        self.scs2 = SharpenedCosineSimilarity(
            in_channels=8, out_channels=16, kernel_size=3, padding=1)
        self.scs3 = SharpenedCosineSimilarity(
            in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.scs4 = SharpenedCosineSimilarity(
            in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.scs5 = SharpenedCosineSimilarity(
            in_channels=64, out_channels=128, kernel_size=3, padding=1)
        
        # CHANGE out_features = 2 or 3 based on number of classes
        self.out = nn.Linear(in_features=6272, out_features=2) # 128*7*7 = 6272

        self.c1 = nn.Conv2d(3,8,3,1,0)
        self.c2 = nn.Conv2d(8,16,3,1,1)
        self.c3 = nn.Conv2d(16,32,3,1,1)
        self.relu = nn.ReLU()

        self.senet = SENet(32,16)

  def forward(self, t):
      tc = self.relu(self.c1(t))
      t = self.scs1(t)

      t = torch.mul(t,tc)

      t = self.pool(t)
      tc = self.pool(tc)

      tc = self.relu(self.c2(tc))      
      t = self.scs2(t)

      t = torch.mul(t,tc)

      t = self.pool(t)
      tc = self.pool(tc)

      tc = self.relu(self.c3(tc))
      t = self.scs3(t)

      t = torch.mul(t,tc)

      t = self.pool(t)

      t = self.senet(t)

      t = self.scs4(t)
      t = self.pool(t)

      t = self.scs5(t)
      t = self.pool(t)

      t=t.flatten(start_dim=1)

      t = self.out(t)

      return t