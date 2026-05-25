import torch
import torch.nn as nn
import torch.nn.functional as F

from models.backbones import backbones
from models.classifiers import Class_Balanced_Softmax
from utils.compute import NormedLinear
from utils.meter import AverageMeter


def scl_loss_func(anchor_features, contrast_features, anchor_targets, contrast_targets, num_classes, temperature):
    device = (torch.device('cuda')
              if anchor_features.is_cuda
              else torch.device('cpu'))

    mask = torch.eq(anchor_targets, contrast_targets.T).float().to(device)
    logits_mask = torch.scatter(
        torch.ones_like(mask),
        1,
        torch.arange(anchor_features.shape[0]).view(-1, 1).to(device),
        0
    )
    mask = mask * logits_mask  # the mask for identifying same class sample, while excluding self-contrastive.

    # class-complement
    logits = anchor_features.mm(contrast_features.T)
    logits = torch.div(logits, temperature)

    # For numerical stability
    logits_max, _ = torch.max(logits, dim=1, keepdim=True)
    logits = logits - logits_max.detach()

    # class-averaging
    exp_logits = torch.exp(logits) * logits_mask

    batch_cls_count = torch.eye(num_classes).to(anchor_targets.device)[contrast_targets].sum(dim=0).squeeze()
    per_ins_weight = torch.tensor([batch_cls_count[i] for i in contrast_targets], device=device).view(1, -1).expand(
        mask.shape[0], mask.shape[1]) - mask
    exp_logits_sum = exp_logits.div(per_ins_weight).sum(dim=1, keepdim=True)

    log_prob = logits - torch.log(exp_logits_sum)
    mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

    loss = - mean_log_prob_pos
    loss = loss.mean()
    return loss


class BCLModel(nn.Module):
    def __init__(self, backbone, use_norm=True, feat_dim=1024, hidden_dim=None, dataset=None, temperature=0.07,
                 alpha=1.0, beta=0.35, loss=None, classifier=None):
        super(BCLModel, self).__init__()
        self.backbone = backbones[backbone['name']](**backbone['args'])
        num_features = self.backbone.inplanes
        if hidden_dim is None:
            hidden_dim = num_features
        self.head = nn.Sequential(nn.Linear(num_features, hidden_dim),
                                  nn.BatchNorm1d(hidden_dim),
                                  nn.ReLU(inplace=True),
                                  nn.Linear(hidden_dim, feat_dim))
        if use_norm:
            self.fc = NormedLinear(num_features, dataset.num_classes, s=30)
        else:
            self.fc = nn.Linear(num_features, dataset.num_classes)

        self.criterion_ce = Class_Balanced_Softmax(dataset=dataset, fc=False, loss=loss, **classifier['args'])
        self.num_classes = dataset.num_classes
        self.dataset = dataset
        self.temperature = temperature
        self.alpha = alpha
        self.beta = beta
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.loss_ce = AverageMeter('Loss_C', ':.3f')
        self.loss_scl = AverageMeter('Loss_S', ':.3f')

        self.head_center = nn.Sequential(nn.Linear(num_features, hidden_dim),
                                         nn.BatchNorm1d(hidden_dim),
                                         nn.ReLU(inplace=True),
                                         nn.Linear(hidden_dim, feat_dim))
        self.register_buffer("targets_centers", torch.arange(self.num_classes).view(-1, 1))
        self.check_grad = True

    def pre_epoch_setup(self, epoch, device):
        self.loss_ce.reset()
        self.loss_scl.reset()

    def get_loss_components(self):
        return [self.loss_ce, self.loss_scl]

    def to(self, device):
        super().to(device)
        self.criterion_ce.to(device)

    def forward(self, x, targets=None):
        feat = self.backbone(x)

        if self.training:
            batch_size = feat.shape[0] // 3
            targets, anchor_targets = torch.split(targets, [batch_size, batch_size + batch_size], dim=0)
            f1, f2 = torch.split(feat, [batch_size, batch_size + batch_size], dim=0)

            logits = self.fc(f1)
            ce_loss = self.criterion_ce(logits=logits, targets=targets)['loss']

            anchor_targets = anchor_targets.contiguous().view(-1, 1)
            anchor_features = F.normalize(self.head(f2), dim=1)
            centers = F.normalize(self.head_center(self.fc.weight.T), dim=1)
            centers = centers[:self.num_classes]

            contrast_targets = torch.cat([anchor_targets, self.targets_centers], dim=0)
            contrast_features = torch.cat([anchor_features, centers], dim=0)
            scl_loss = scl_loss_func(anchor_features, contrast_features, anchor_targets, contrast_targets,
                                     self.num_classes, self.temperature)

            loss = self.alpha * ce_loss + self.beta * scl_loss

            self.loss_ce.update(ce_loss.item())
            self.loss_scl.update(scl_loss.item())

            output = {"similarities": logits, "loss": loss, "targets": targets}
            if self.check_grad:
                anchor_features.retain_grad()
                output.update({"anchor_features": anchor_features,
                               "contrast_features": contrast_features,
                               "anchor_targets": anchor_targets,
                               "contrast_targets": contrast_targets,
                               "beta": self.beta}
                              )
            return output
        else:
            logits = self.fc(feat)
            return {"similarities": logits}


def bcl_model(**kwargs):
    return BCLModel(**kwargs)
