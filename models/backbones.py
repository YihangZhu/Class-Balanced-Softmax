from models import resnet_cifar
from models import resnet_imagenet

backbones = {}
backbones.update(resnet_cifar.backbones)
backbones.update(resnet_imagenet.backbones)
