import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.meter import AverageMeter
from utils.utils import IsNewTrainEpoch


class BinaryLogisticLoss(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.bce = nn.functional.binary_cross_entropy

    def forward(self, sigmoid_feature, target_codes):
        # sigmoid_feature_zero = torch.where(sigmoid_feature > 0.5, sigmoid_feature, 0)
        # sigmoid_feature_one = torch.where(sigmoid_feature <= 0.5, sigmoid_feature, 1)
        # sigmoid_feature_select = torch.where(target_codes == 1, sigmoid_feature_one, sigmoid_feature_zero)
        loss = self.bce(sigmoid_feature, target_codes, reduction='mean')
        return loss


class MSELoss(nn.Module):
    def __init__(self, reduction, **kwargs):
        super().__init__()
        self.mse = nn.MSELoss(reduction=reduction)

    def forward(self, feature, target_feature):
        loss = self.mse(feature, target_feature)
        return loss


class FocalBCE(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.bce = nn.functional.binary_cross_entropy

    def forward(self, sigmoid_feature, target_codes):
        mask_one = sigmoid_feature <= 0.5
        mask_one *= target_codes > 0.9
        mask_zero = sigmoid_feature > 0.5
        mask_zero *= target_codes < 0.1
        mask = mask_zero + mask_one

        def get_loss_components():
            selected_sigmoid_feature = sigmoid_feature[mask]
            selected_target_code = target_codes[mask]
            return self.bce(selected_sigmoid_feature, selected_target_code, reduction='mean')

        loss1 = get_loss_components()
        mask = torch.logical_not(mask)
        loss2 = get_loss_components()
        loss = loss1 + loss2
        return loss


class CrossEntropyLoss(nn.Module):
    def __init__(self, label_smoothing=0.0, **kwargs):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    def forward(self, output_logits, target):
        loss = self.ce(output_logits, target)
        return loss


def _focal_loss(input_values, gamma, detach):
    """Computes the focal loss"""
    p = torch.exp(-input_values)
    if detach:
        p = p.detach()
    loss = (1 - p) ** gamma * input_values
    return loss.mean()


class FocalLoss(nn.Module):
    def __init__(self, ce_type='ce', reduction='none', weight=None, gamma=2, **kwargs):
        super().__init__()
        if ce_type == 'logit_bce':
            self.func = F.binary_cross_entropy_with_logits
        elif ce_type == 'ce':
            self.func = F.cross_entropy

        self.reduction = reduction
        self.gamma = gamma
        self.weight = weight

    def forward(self, inputs, targets):
        bce = self.func(inputs, targets, reduction=self.reduction, weight=self.weight)
        loss = _focal_loss(bce, self.gamma, detach=False)
        return loss


def dice_loss(inputs, targets, smooth=1):
    inputs = torch.sigmoid(inputs)
    intersection = (inputs * targets).sum((1, 2))
    dice = (2. * intersection + smooth) / (inputs.sum((1, 2)) + targets.sum((1, 2)) + smooth)
    loss = (1 - dice).mean()
    return loss


class DRW:
    def __init__(self, dataset, reweight_epoch):
        betas = [0, 0.9999]
        self.weights = []
        cls_num_list = dataset.train_num_per_cls_dict
        for beta in betas:
            effective_num = 1.0 - np.power(beta, cls_num_list)
            per_cls_weights = (1.0 - beta) / np.array(effective_num)
            per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
            self.weights.append(torch.FloatTensor(per_cls_weights))
        self.reweight_epoch = reweight_epoch

    def to(self, device):
        for i, weight in enumerate(self.weights):
            self.weights[i] = weight.to(device)

    def get_weights(self, epoch):
        if epoch >= self.reweight_epoch:
            return self.weights[1]
        else:
            return self.weights[0]


class LDAMLoss(nn.Module):
    def __init__(self, dataset, max_m, s, epoch=0, drw=None, reweight_epoch=None):
        super().__init__()
        cls_num_list = dataset.train_num_per_cls_dict
        m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list))
        m_list = m_list * (max_m / np.max(m_list))
        self.m_list = torch.FloatTensor(m_list)
        assert s > 0
        self.s = s
        if drw is not None:
            assert reweight_epoch is not None
            self.drw = DRW(dataset, reweight_epoch)
        self.is_new_epoch = IsNewTrainEpoch(epoch)

    def setup_start_epoch(self, epoch):
        self.is_new_epoch.set_epoch(epoch)

    def to(self, device):
        super().to(device)
        self.m_list = self.m_list.to(device)
        if hasattr(self, 'drw'):
            self.drw.to(device)

    def forward(self, logit, target):
        self.is_new_epoch()
        mask = torch.zeros_like(logit, dtype=torch.bool)
        mask.scatter_(1, target.data.view(-1, 1), 1)

        mask_float = mask.type(torch.FloatTensor)
        mask_float = mask_float.to(logit.device)
        batch_m = torch.matmul(self.m_list[None, :], mask_float.transpose(0, 1))
        batch_m = batch_m.view((-1, 1))
        x_m = logit - batch_m
        # print(logit[0,target[0]] - batch_m)
        output = torch.where(mask, x_m, logit)
        # print(output[0,target[0]])
        if hasattr(self, 'drw'):
            weights = self.drw.get_weights(self.is_new_epoch.epoch)
        else:
            weights = None
        return F.cross_entropy(self.s * output, target, weight=weights)


def get_ride_temperature(cls_num_list, reweight_factor):
    if reweight_factor is None:
        return [1.0] * len(cls_num_list)
    cls_num_list = np.array(cls_num_list) / np.sum(cls_num_list)
    C = len(cls_num_list)
    per_cls_weights = C * cls_num_list * reweight_factor + 1 - reweight_factor
    # Experimental normalization: This is for easier hyperparameter tuning,
    # the effect can be described in the learning rate so the math formulation keeps the same.
    # At the same time, the 1 - max trick that was previously used is not required since weights are already adjusted.
    per_cls_weights = per_cls_weights / np.max(per_cls_weights)
    assert np.all(per_cls_weights > 0), "reweight factor is too large: out of bounds"
    return per_cls_weights


class RIDELoss(nn.Module):
    def __init__(self, dataset, ldam=None, drw=None, max_m=None, s=None,
                 additional_diversity_factor=None,
                 reweight_factor=None, reweight_epoch=None, epoch=0):
        super().__init__()
        self.additional_diversity_factor = additional_diversity_factor

        if ldam:
            assert s is not None
            self.s = s
            self.base_loss = LDAMLoss(dataset, max_m, s, drw=drw, reweight_epoch=reweight_epoch)
        else:
            self.base_loss = CrossEntropyLoss()

        if self.consider_peer_loss():
            self.reweight_epoch = reweight_epoch
            if self.reweight_epoch is not None:
                self.per_cls_diversity_temperatures = torch.FloatTensor(
                    get_ride_temperature(dataset.train_num_per_cls_dict, reweight_factor)
                ).view((1, -1))
                self.per_cls_diversity_temperatures_mean = self.per_cls_diversity_temperatures.mean().item()

            self.classification_loss = AverageMeter('C_Loss', ':.3f')
            self.diversity_loss = AverageMeter('D_Loss', ':.3f')
            self.classification_loss_values = []
            self.diversity_loss_values = []
        self.is_new_epoch = IsNewTrainEpoch(epoch)

    def get_loss_components(self):
        if self.consider_peer_loss():
            return [self.classification_loss, self.diversity_loss]
        else:
            return None

    def setup_start_epoch(self, epoch):
        self.is_new_epoch.set_epoch(epoch)
        if isinstance(self.base_loss, LDAMLoss):
            self.base_loss.setup_start_epoch(epoch)

    def consider_peer_loss(self):
        return self.additional_diversity_factor is not None

    def get_diversity_loss_weights(self):
        if self.reweight_epoch is not None:
            if self.is_new_epoch.epoch >= self.reweight_epoch:
                return self.per_cls_diversity_temperatures, self.per_cls_diversity_temperatures_mean
        return 1.0, 1.0

    def to(self, device):
        super().to(device)
        self.base_loss.to(device)
        if hasattr(self, 'per_cls_diversity_temperatures'):
            self.per_cls_diversity_temperatures = self.per_cls_diversity_temperatures.to(device)

    def forward(self, logits, logits_mean, target):
        if not self.consider_peer_loss():
            return self.base_loss(logits_mean, target)
        if self.is_new_epoch():
            if self.consider_peer_loss():
                self.classification_loss.reset()
                self.diversity_loss.reset()

        classification_loss = torch.tensor(0.0, device=logits.device)
        diversity_loss = torch.tensor(0.0, device=logits.device, dtype=torch.float)
        # individual classification loss
        logits = logits.transpose(0, 1)
        for logits_item in logits:
            classification_loss += self.base_loss(logits_item, target)

        # diversity loss, if using LDAM for base loss, s is incorporated in LDAM,
        # therefore, we also multiply logits with s according to the published code of RIDE.
        if self.additional_diversity_factor != 0:
            if hasattr(self, 's'):
                logits *= self.s
                logits_mean *= self.s

            diversity_temperature, temperature_mean = self.get_diversity_loss_weights()
            with torch.no_grad():
                # Using the mean takes only linear instead of quadratic time in computing and
                # has only a slight difference so using the mean is preferred here
                mean_output_dist = F.softmax(logits_mean / diversity_temperature, dim=1)
            for logits_item in logits:
                output_dist = F.log_softmax(logits_item / diversity_temperature, dim=1)
                diversity_loss += self.additional_diversity_factor * temperature_mean * temperature_mean * F.kl_div(
                    output_dist, mean_output_dist, reduction='batchmean')
        if self.consider_peer_loss():
            self.classification_loss.update(classification_loss)
            self.diversity_loss.update(diversity_loss)
        return classification_loss + diversity_loss, logits_mean


# class BalancedSoftmaxLoss(nn.Module):
#     """
#     Balanced Softmax Loss
#     """
#
#     def __init__(self, dataset):
#         super().__init__()
#         self.num_per_cls_dict = torch.tensor(dataset.train_num_per_cls_dict, dtype=torch.float)
#
#     def to(self, device):
#         super().to(device)
#         self.num_per_cls_dict = self.num_per_cls_dict.to(device)
#
#     def forward(self, x, targets):
#         return self.balanced_softmax_loss(targets, x)
#
#     def balanced_softmax_loss(self, targets, logits, reduction='mean'):
#         """Compute the Balanced Softmax Loss between `logits` and the ground truth `labels`.
#         Args:
#           targets: A int tensor of size [batch].
#           logits: A float tensor of size [batch, no_of_classes].
#           reduction: string. One of "none", "mean", "sum"
#         Returns:
#           loss: A float tensor. Balanced Softmax Loss.
#         """
#         spc = self.num_per_cls_dict.type_as(logits)
#         spc = spc.unsqueeze(0).expand(logits.shape[0], -1)
#         balanced_logits = logits + spc.log()
#         loss = F.cross_entropy(input=balanced_logits, target=targets, reduction=reduction)
#         return loss


# class ImprovedBalancedSoftmaxLoss(BalancedSoftmaxLoss):
#     def __init__(self, dataset, gamma=None):
#         super().__init__(dataset)
#         # p = torch.softmax(torch.log(self.num_per_cls_dict), dim=0)
#         if gamma is None:
#             self.num_per_cls_dict = torch.tensor(dataset.class_prob)
#         else:
#             self.num_per_cls_dict = torch.pow(self.num_per_cls_dict, (1.0 + gamma) / gamma)
#         # p2 = torch.softmax(torch.log(self.num_per_cls_dict), dim=0)
#         # import matplotlib.pyplot as plt
#         # print(p)
#         # print(p2)
#         # plt.plot(p, label="bs")
#         # plt.plot(p2, label="ibs")
#         # plt.legend()
#         # plt.show()
#         # exit()

class SIoULoss(nn.Module):
    def __init__(self, eps=1e-7):
        super(SIoULoss, self).__init__()
        self.eps = eps

    def forward(self, pred_boxes, target_boxes):
        """
        pred_boxes and target_boxes are in format (cx, cy, w, h)
        """

        px, py, pw, ph = pred_boxes[:, 0], pred_boxes[:, 1], pred_boxes[:, 2], pred_boxes[:, 3]
        tx, ty, tw, th = target_boxes[:, 0], target_boxes[:, 1], target_boxes[:, 2], target_boxes[:, 3]

        # Convert (cx, cy, w, h) -> (x1, y1, x2, y2)
        pred_x1 = px - pw / 2
        pred_y1 = py - ph / 2
        pred_x2 = px + pw / 2
        pred_y2 = py + ph / 2

        target_x1 = tx - tw / 2
        target_y1 = ty - th / 2
        target_x2 = tx + tw / 2
        target_y2 = ty + th / 2

        # Intersection box
        inter_x1 = torch.max(pred_x1, target_x1)
        inter_y1 = torch.max(pred_y1, target_y1)
        inter_x2 = torch.min(pred_x2, target_x2)
        inter_y2 = torch.min(pred_y2, target_y2)

        inter_area = (inter_x2 - inter_x1).clamp(0) * (inter_y2 - inter_y1).clamp(0)
        pred_area = (pred_x2 - pred_x1) * (pred_y2 - pred_y1)
        target_area = (target_x2 - target_x1) * (target_y2 - target_y1)
        union_area = pred_area + target_area - inter_area + self.eps

        iou = inter_area / union_area

        # ----------------- SIoU Components -----------------

        # Distance cost
        cx_dist = (tx - px)
        cy_dist = (ty - py)

        sigma = torch.sqrt(cx_dist ** 2 + cy_dist ** 2 + self.eps)

        # Angle cost
        sin_alpha = torch.abs(cx_dist) / (sigma + self.eps)
        angle_cost = torch.cos(torch.arcsin(sin_alpha) * 2 - torch.pi / 2)

        # Distance cost with angle
        rho_x = (cx_dist / (tw + self.eps)) ** 2
        rho_y = (cy_dist / (th + self.eps)) ** 2
        gamma = angle_cost - 2

        distance_cost = 2 - torch.exp(-gamma * rho_x) - torch.exp(-gamma * rho_y)

        # Shape cost
        omega_w = torch.abs(pw - tw) / torch.max(pw, tw)
        omega_h = torch.abs(ph - th) / torch.max(ph, th)
        shape_cost = torch.pow(1 - torch.exp(-omega_w), 4) + torch.pow(1 - torch.exp(-omega_h), 4)

        siou_loss = 1 - iou + 0.5 * (distance_cost + shape_cost)

        return siou_loss.mean()
