from collections import OrderedDict
import torch
from torchvision import models
from torch.utils import model_zoo
import torch.nn as nn


class ResNetSIN(nn.Module):
    url = 'https://bitbucket.org/robert_geirhos/texture-vs-shape-pretrained-models/raw/6f41d2e86fc60566f78de64ecff35cc61eb6436f/resnet50_train_60_epochs-c8e5653e.pth.tar'

    def __init__(self, num_classes=1000,device='cpu'):
        super().__init__()
        # Load pretrained model of ResNet on Stylized ImageNet (Resnet_SIN)
        self.resnet_sin = models.resnet50(weights=None, num_classes=num_classes)
        checkpoint = model_zoo.load_url(self.url, map_location=torch.device('cpu'))

        # Remove prefix module. for all keys in checkpoint['state_dict']
        new_state_dict = OrderedDict()
        for k, v in checkpoint['state_dict'].items():
            name = k[7:]  # remove `module.`
            new_state_dict[name] = v

        # Load the new state dict
        self. resnet_sin.load_state_dict(new_state_dict)

        # Change the last layer to output num_classes
        # self.resnet_sin.eval()
        self.resnet_sin.to(device)


    def forward(self, x):
        return self.resnet_sin(x)
