import logging

import torch.nn as nn
from torchvision.models import vgg16_bn, resnet18, resnet34

from utils.logger import Logger

VGG16 = 'vgg16'
RESNET18 = 'resnet18'
RESNET34 = 'resnet34'

class HABClassifier(nn.Module):
    __names__ = {RESNET18, RESNET34, VGG16}

    def __init__(self, arch, num_classes, pretrained=False):
        super(HABClassifier, self).__init__()

        assert arch in HABClassifier.__names__

        self.pretrained = pretrained
        self.num_class = num_classes
        self.logger = logging.getLogger(__name__)

        if arch == VGG16:
            self.feature_extractor, self.classifier = HABClassifier.get_vgg16_arch(
                num_classes, pretrained)

        elif arch == RESNET18:
            self.feature_extractor, self.classifier = HABClassifier.get_resnet18_arch(
                num_classes, pretrained)

        elif arch == RESNET34:
            self.feature_extractor, self.classifier = HABClassifier.get_resnet34_arch(
                num_classes, pretrained)

        Logger.section_break('Model')
        self.logger.info(f'Architecture selected: {arch} | Pretrained: {pretrained} | '
                         f'Num classes: {num_classes}')

    def forward(self, x):
        x = self.feature_extractor(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def get_params(self):
        return self.parameters()

    @staticmethod
    def get_vgg16_arch(num_classes, pretrained=False):
        vgg16_model = vgg16_bn(pretrained=pretrained)
        feature_extractor = nn.Sequential(*list(vgg16_model.features.children()))

        num_features = vgg16_model.classifier[-1].in_features
        classifier = nn.Sequential(*list(vgg16_model.classifier.children())[:-1],
                                   nn.Linear(num_features, num_classes))

        return feature_extractor, classifier

    @staticmethod
    def get_resnet18_arch(num_classes, pretrained=False):
        resnet18_model = resnet18(pretrained=pretrained)

        feature_extractor = nn.Sequential(*list(resnet18_model.children())[:-1])

        num_features = resnet18_model.fc.in_features
        classifier = nn.Linear(num_features, num_classes)

        return feature_extractor, classifier

    @staticmethod
    def get_resnet34_arch(num_classes, pretrained=False):
        resnet34_model = resnet34(pretrained=pretrained)

        feature_extractor = nn.Sequential(*list(resnet34_model.children())[:-1])

        num_features = resnet34_model.fc.in_features
        classifier = nn.Linear(num_features, num_classes)

        return feature_extractor, classifier


if __name__ == '__main__':
    import torch

    device = torch.device("cpu")
    model = HABClassifier(arch='resnet18', pretrained=True,
                          num_classes=22)
    print(model)

    inp = torch.rand(16, 3, 224, 224)
    inp = inp.to(device)
    print('Input:', inp.shape)
    out = model(inp)
    print('Output:', out.shape)