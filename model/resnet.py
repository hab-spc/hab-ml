""" Create/Modify ResNet models

#TODO model resnet description

"""

import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import model.layers as L
import collections


__all__ = [ 
    'resnet10',
    'resnet18',
    'resnet34',
    'resnet50',
    'resnet10_dropout',
    'resnet18_dropout',
    'resnet34_dropout',
    'resnet50_dropout',
]


class ResNet(nn.Module):
    def __init__(self, block, nb_blocks, nb_channels, downsample, num_classes, dropout=False):
        super(ResNet, self).__init__()
        name = 'conv1'
        module = L.ConvBlock(3, 64, 7, stride=2, padding=3, 
            activation_fcn=F.relu, dropout=False, batch_norm=True)
        self.add_module(name, module)

        self.layers = []
        cur_channels = 64
        for b in range(len(nb_blocks)):
            for bb in range(nb_blocks[b]):
                name = 'block-{}-{}'.format(b+1, bb+1)
                ds = downsample[b] if bb==0 else False
                do = dropout and b==(len(nb_blocks)-1)
                module = block(cur_channels, nb_channels[b], downsample=ds, dropout=do)
                self.layers.append(module)
                self.add_module(name, module)
                cur_channels = nb_channels[b]

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = L.linear(cur_channels, num_classes, use_bias=True)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.normal_(m.weight, mean=0., std=0.01)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        for layer in self.layers:
            x = layer(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def load_pretrained(self, pretrained_fn):
        """ Load pretrained model

        Given an initialized model, one can load a pretrained model using
        this function. It will expect one of the pretrained files from
        pytorch in order to successfully load it.

        Currently, the user must manually download it using the
        `download_pretrained.py`, before being able to load a pretrained model

        Args:
            pretrained_fn (str): Absolute path to the pretrained ResNet model

        Returns:
            None

        """
        checkpoint = torch.load(pretrained_fn)
        state_dict = self.state_dict()
#         print('\n' + '='*30+' State Dict Keys '+'='*30)
#         print(state_dict.keys())
#         print('\n' + '='*30+' Checkpoint Keys '+'='*30)
#         for k in state_dict:
#             print(k, state_dict[k].shape)

        pretrained_fn = collections.OrderedDict()
        for key in state_dict:
            key_tkn = key.split('.')
            if key_tkn[-1] == 'num_batches_tracked':
                continue

            if key_tkn[0] == 'conv1':
                if key_tkn[1] == 'conv':
                    ckp_key = '.'.join([key_tkn[0], key_tkn[2]])
                elif key_tkn[1] == 'bn':
                    ckp_key = '.'.join(['bn1', key_tkn[2]])
                pretrained_fn[key] = checkpoint[ckp_key]

            elif key_tkn[0] == 'classifier':
                pretrained_fn[key] = state_dict[key]

            else:
                layer = key_tkn[0].split('-')[1:]
                if key_tkn[1] == 'conv_res':
                    ckp_key = 'layer{}.{}.downsample.0.{}'.format(layer[0], int(layer[1])-1, key_tkn[-1])
                elif key_tkn[1] == 'bn_res':
                    ckp_key = 'layer{}.{}.downsample.1.{}'.format(layer[0], int(layer[1])-1, key_tkn[-1])
                else:
                    ckp_key = 'layer{}.{}.'.format(layer[0], int(layer[1])-1) + '.'.join(key_tkn[1:])
                pretrained_fn[key] = checkpoint[ckp_key]

        self.load_state_dict(pretrained_fn)

"""BEGIN: ResNet model functions"""
def resnet10(num_classes=1000):
    return ResNet(L.BasicResBlock, [1, 1, 1, 1],  [64, 128, 256, 512], [False, True, True, True], num_classes)

def resnet18(num_classes=1000):
    return ResNet(L.BasicResBlock, [2, 2, 2, 2],  [64, 128, 256, 512], [False, True, True, True], num_classes)

def resnet34(num_classes=1000):
    return ResNet(L.BasicResBlock, [3, 4, 6, 3],  [64, 128, 256, 512], [False, True, True, True], num_classes)

def resnet50(num_classes=1000):
    return ResNet(L.BottleneckResBlock, [3, 4, 6, 3],  [64*4, 128*4, 256*4, 512*4], [False, True, True, True], num_classes)

def resnet10_dropout(num_classes=1000):
    return ResNet(L.BasicResBlock, [1, 1, 1, 1],  [64, 128, 256, 512], [False, True, True, True], num_classes, dropout=True)

def resnet18_dropout(num_classes=1000):
    return ResNet(L.BasicResBlock, [2, 2, 2, 2],  [64, 128, 256, 512], [False, True, True, True], num_classes, dropout=True)

def resnet34_dropout(num_classes=1000):
    return ResNet(L.BasicResBlock, [3, 4, 6, 3],  [64, 128, 256, 512], [False, True, True, True], num_classes, dropout=True)

def resnet50_dropout(num_classes=1000):
    return ResNet(L.BottleneckResBlock, [3, 4, 6, 3],  [64*4, 128*4, 256*4, 512*4], [False, True, True, True], num_classes, dropout=True)
"""END: ResNet model functions"""


def create_model(arch, num_classes=1000):
    """ Create ResNet Model

    Args:
        arch (str): Desired ResNet version
        num_classes (int): Number of classes

    Returns:
        resnet_fn(): Pointer to ResNet model function intialization

    """
    assert arch in __all__
    # Evaluates string into function representation e.g. `resnet18_dropout()`
    resnet_fcn = eval(arch)
    return resnet_fcn(num_classes)


if __name__ == '__main__':
    """Example here of how to initialize model"""
    device = torch.device("cpu")
    model = create_model('resnet50', num_classes=2)
    print(model)
    
    inp = torch.rand(16, 3, 112, 112)
    inp = inp.to(device)
    print('Input:', inp.shape)
    out = model(inp)
    print('Output:', out.shape)
    
    fn = 'resnet50.pth'
    if not os.path.exists(fn):
        os.system('wget -O resnet50.pth https://download.pytorch.org/models/resnet50-19c8e357.pth')
    model.load_pretrained(fn)
    print('Loaded imagenet pretrained checkpoint {}'.format(fn))
    for ct, child in enumerate(model.children()):
        if ct < 9:
            for param in child.parameters():
                param.requires_grad = False
        print(ct, child, param.requires_grad)