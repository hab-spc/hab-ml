"""Download pretrained models"""
# Standard dist imports
import os

# ResNet18, Resnet34, ResNet50
os.system('wget -O resnet18.pth https://download.pytorch.org/models/resnet18-5c106cde.pth')
os.system('wget -O resnet34.pth https://download.pytorch.org/models/resnet34-333f7ec4.pth')
os.system('wget -O resnet50.pth https://download.pytorch.org/models/resnet50-19c8e357.pth')
