from torch import nn
from torchvision import models

class BaseNet(nn.Module):
    def __init__(self,name='resnet50'):
        super().__init__()
        backbones={'resnet50':models.resnet50}
        self.name=name
        self.basenet=nn.Sequential(*list(backbones[self.name](weights="IMAGENET1K_V1").children())[:-1])
    def forward(self,x):
        x=self.basenet(x).reshape(1,-1)
        return x
