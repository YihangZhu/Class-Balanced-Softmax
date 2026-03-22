import torch
import torch.nn as nn

from models import classifiers as classifier_models
from models import resnet_cifar
from models import resnet_imagenet
from models.backbones import backbones


class LinearMerge(nn.Module):
    def __init__(self, num_channels, num_parameters, require_grad):
        super().__init__()
        self.linears = nn.Parameter(torch.zeros((num_channels, num_parameters)), requires_grad=require_grad)
        self.linears.data.fill_(1.0 / num_parameters)
        # with torch.no_grad():
        #     for i in range(num_channels):
        #         # following the default way in torch init for linear layer weight.
        #         # print(self.linears.data[i,:])
        #         self.linears.data[i, :].uniform_(-1 / math.sqrt(num_parameters), 1 / math.sqrt(num_parameters))
        #         # print(self.linears.data[i, :])

    def forward(self, x):
        # print(f"x device: {x.device}, linear device: {self.linears[0].weight.device}")
        # output = []
        output = torch.mul(x.view((x.shape[0], x.shape[1], -1)), self.linears)
        output = output.sum(dim=-1)
        # for i in range(x.shape[1]):
        #     x_reshape = x[:, i].view((x.shape[0], -1))
        #     x_out = self.linears[i](x_reshape)
        #     output.append(x_out)
        # output = torch.stack(output, dim=1)
        return output


class MergeLayer(nn.Module):
    def __init__(self, num_channels, num_features, kernel_size, require_grad, mix_channels):
        super().__init__()
        self.merge_layer = nn.Sequential()
        if num_channels > num_features or mix_channels:
            self.merge_layer.append(nn.Conv2d(num_channels, num_features, kernel_size=1, bias=False))
        self.merge_layer.append(LinearMerge(num_features, kernel_size ** 2, require_grad))

    def forward(self, x):
        x = self.merge_layer(x)
        return x


def make_classifier(dataset, classifier, loss, representation_model=None, ):
    if 'num_features' not in classifier['args']:
        classifier['args']['num_features'] = representation_model.inplanes

    classifier = getattr(classifier_models, classifier['name'])(dataset=dataset,
                                                                **classifier['args'],
                                                                loss=loss)
    return classifier


class LearningModel(nn.Module):
    def __init__(self, backbone, classifier, dataset=None, loss=None):
        super().__init__()
        backbone_func = backbones[backbone['name']]
        if 'vit' in backbone['name']:
            backbone['args']['image_size'] = dataset.image_size

        self.representation_model = backbone_func(**backbone['args'])

        self.classifier = make_classifier(dataset, classifier, loss, self.representation_model)

        if isinstance(self.representation_model, resnet_cifar.ResNet_Cifar):
            self.apply(resnet_cifar.weights_init)
        elif isinstance(self.representation_model, resnet_imagenet.ResNet):
            resnet_imagenet.init_weight(self)
        self.check_grad = False

    def to(self, device):
        super().to(device)
        self.representation_model.to(device)
        self.classifier.to(device)

    def forward(self, x, targets=None):
        features = self.representation_model(x)
        output = self.classifier(features, targets)
        output['feature'] = features
        if self.check_grad:
            output['similarities'].retain_grad()
            output['features'].retain_grad()

        return output

    def set_check_grad(self):
        self.check_grad = True
