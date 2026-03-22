import torch
import torch.nn as nn

from models import losses
from utils.compute import NormedLinear
from utils.compute import normalize

__all__ = ['Standard_Classifier', 'Post_Adjust', 'Class_Balanced_Softmax']


class Standard_Classifier(nn.Module):
    def __init__(self, dataset, num_classes=None, num_features=None, linear_bias=True, fc=True, cos_similarity=False,
                 s=1.0, loss=None):
        super(Standard_Classifier, self).__init__()
        if dataset is not None:
            num_classes = dataset.num_classes
        if fc:
            if cos_similarity:
                self.fc = NormedLinear(num_features, num_classes, s=s)
            else:
                self.fc = nn.Linear(num_features, num_classes, bias=linear_bias)
        if loss is not None:
            self.criterion = getattr(losses, loss['name'])(**loss['args'])
        self.num_per_cls_dict = torch.tensor(
            dataset.train_num_per_cls_dict, dtype=torch.float
        ) if dataset is not None and dataset.train_num_per_cls_dict is not None else None

    def to(self, device):
        if self.num_per_cls_dict is not None:
            self.num_per_cls_dict = self.num_per_cls_dict.to(device)

    def _train(self, output):
        if hasattr(self, 'fc'):
            output['similarities'] = self.fc(output['features'])
        return output

    def _get_loss(self, output):
        output['loss'] = self.criterion(output['similarities'], output['targets'])
        return output

    def _inference(self, output):
        if hasattr(self, 'fc'):
            output['similarities'] = self.fc(output['features'])
        return output

    def forward(self, features=None, targets=None, logits=None):
        output = {'features': features, 'similarities': logits, 'targets': targets}
        if self.training:
            output = self._train(output)
            if hasattr(self, 'criterion'):
                output = self._get_loss(output)
        else:
            output = self._inference(output)
        return output

    def get_linear_weights_bias(self):
        if isinstance(self.fc, nn.Linear):
            return self.fc.weight, self.fc.bias
        else:
            return None, None


class Post_Adjust(Standard_Classifier):
    def __init__(self, post_adjust=None, **kwargs):
        super(Post_Adjust, self).__init__(**kwargs)
        self.adjust_func = getattr(self, post_adjust)

    def _inference(self, output):
        return self.adjust_func(output)

    def tau_norm(self, output):
        weight = normalize(self.fc.weight.detach(), power=self.tau_norm)
        output['similarities'] = torch.mm(output['feature'], torch.t(weight))
        return output

    def tau_norm_both(self, output):
        feature_normalized = normalize(output['feature'].detach(), power=self.tau_norm_both)
        weight_normalized = normalize(self.fc.weight.detach(), power=self.tau_norm_both)
        output['similarities'] = torch.mm(feature_normalized, torch.t(weight_normalized))
        return output

    def post_hoc_adjust_logit(self, output):
        output = super()._inference(output)
        spc = self.num_per_cls_dict.type_as(output['similarities'])
        spc = spc.unsqueeze(0).expand(output['similarities'].shape[0], -1)
        output['similarities'] = output['similarities'] - spc.log()
        return output


class Class_Balanced_Softmax(Standard_Classifier):
    def __init__(self, power_law, **kwargs):
        super().__init__(**kwargs)

        if isinstance(power_law, str):
            power_law = eval(power_law)
        self.weights = (
            torch.pow(self.num_per_cls_dict, power_law)
        ).log()

        # elif crc is not None:
        #     self.crc = crc
        # elif cosine is not None:
        #     s = max(self.num_per_cls_dict) / (torch.pi / 2)
        #     self.weights = (
        #         torch.pow(
        #             torch.cos(self.num_per_cls_dict / s + 3 / 2 * torch.pi),
        #             cosine
        #         )
        #     ).log()

        # elif logarithm is not None:
        #     max_n = max(self.num_per_cls_dict)
        #     min_n = min(self.num_per_cls_dict)
        #
        #     self.weights = (
        #         torch.pow(
        #             torch.log(1 + (self.num_per_cls_dict - min_n + 1) / (max_n - min_n + 1) * (torch.e - 1)),
        #             logarithm
        #         )
        #     ).log()

    def to(self, device):
        super().to(device)
        self.weights = self.weights.to(device)

    def _train(self, output):
        output = super()._train(output)
        logit = output['similarities']
        output['similarity_b'] = logit.detach()

        if hasattr(self, 'crc'):
            a = (1.0 / self.num_per_cls_dict[output['targets']]).unsqueeze(1)
            beta = torch.mm(a, self.num_per_cls_dict.unsqueeze(0))
            beta[logit < 0] = 1.0 / beta[logit < 0]
            beta = beta ** self.crc
            logit = logit * beta
            output['grad_beta'] = beta
        output['similarities'] = logit + self.weights
        return output
