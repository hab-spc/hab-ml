import logging

import torch.nn as nn
from torchvision.models import alexnet

class DeepCORAL(nn.Module):
    def __init__(self, num_classes=1000):
        super(DeepCORAL, self).__init__()
        self.shared_net = alexnet(pretrained=True)
        self.shared_net.classifier[6] = nn.Linear(4096, num_classes)

        nn.init.normal_(self.shared_net.classifier[6].weight, mean=0, std=5e-3)
        self.shared_net.classifier[6].bias.data.fill_(0.01)

    def forward(self, source, target):
        source = self.shared_net(source)

        target = self.shared_net(target)
        return source, target
