import logging

import torch.nn as nn
from torchvision.models import alexnet, resnet18

from utils.config import opt

ALEXNET = 'alexnet'
VGG16 = 'vgg16'
RESNET18 = 'resnet18'

class HABCORAL(nn.Module):
    __names__ = {ALEXNET, RESNET18, VGG16}

    def __init__(self, arch, num_classes=1000, coral=opt.coral):
        super(HABCORAL, self).__init__()
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        self.logger.debug('Initialized model network')
        self.logger.info(f'Architecture selected: {arch} | Num Classes: {num_classes}')

        self.num_classes = num_classes
        self.coral = coral
        if arch == ALEXNET:
            self.shared_net, self.classifier = HABCORAL.get_alexnet_arch(num_classes)
            if self.coral:
                self.classifier[6].weight.data.normal_(0, 0.005)
                self.classifier[6].bias.data.fill_(0.01)


        elif arch == RESNET18:
            self.shared_net, self.classifier = HABCORAL.get_resnet18_arch(num_classes)
            if self.coral:
                self.classifier.weight.data.normal_(0, 0.005)
                self.classifier.bias.data.fill_(0.01)

        self.logger.debug(self.shared_net)

    def forward(self, source, target):
        src = self.shared_net(source)
        src = src.view(src.size(0), -1)
        src = self.classifier(src)

        tgt = self.shared_net(target)
        tgt = tgt.view(tgt.size(0), -1)
        tgt = self.classifier(tgt)
        return src, tgt

    def get_params(self):
        if self.coral:
            parameters = [
                {'params': self.shared_net.parameters()},
                {'params': self.classifier.parameters(), 'lr': 10 * opt.lr}]
            self.logger.debug('Parameters returned: {}'.format(parameters))
            return parameters
        else:
            return self.parameters()

    @staticmethod
    def get_alexnet_arch(num_classes, pretrained=False):
        # feature_extractor = AlexNet()
        alexnet_model = alexnet(pretrained=pretrained)

        feature_extractor = alexnet_model.features

        classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

        return feature_extractor, classifier

    @staticmethod
    def get_resnet18_arch(num_classes, pretrained=False):
        resnet18_model = resnet18(pretrained=pretrained)

        feature_extractor = nn.Sequential(*list(resnet18_model.children())[:-1])

        num_features = resnet18_model.fc.in_features
        classifier = nn.Linear(num_features, num_classes)

        return feature_extractor, classifier
