from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms
from torchvision import transforms

from utils.meter import get_dominated_count
from utils.utils import check_file, check_identical


def get_encoded_feature(feature, encode_type):
    encoded_feature = None
    if 'sigmoid' in encode_type:
        sigmoid_feature = torch.sigmoid(feature)
        if encode_type == 'sigmoid_binary':
            encoded_feature = torch.where(sigmoid_feature > 0.5, 1.0, 0.0)
        elif encode_type == 'sigmoid_semi_binary':
            encoded_feature = torch.where(sigmoid_feature > 0.5, sigmoid_feature, 0.0)
        return sigmoid_feature, encoded_feature
    elif 'tanh' in encode_type:
        tanh_feature = torch.tanh(feature)
        if encode_type == 'tanh_binary':
            encoded_feature = torch.where(tanh_feature > 0.5, 1.0, 0.0)
        elif encode_type == 'tanh_semi_binary':
            encoded_feature = torch.where(tanh_feature > 0.5, tanh_feature, 0.0)
        return tanh_feature, encoded_feature


def modify_code(code):
    idx = np.random.randint(len(code))
    new_code = code[:idx] + str(1 - int(code[idx]))
    if idx < len(code) - 1:
        new_code += code[idx + 1:]
    return new_code


# def process_class_mean_code(class_mean_code):
#     codes_in_binary = []
#     for i, code in enumerate(class_mean_code):
#         # replace all non-digital with ""
#         code = code.detach().cpu().numpy()
#         binary_code = re.sub("\D", "", str(code))
#         # print(binary_code)
#         codes_in_binary.append(binary_code)
#
#     counts = Counter(codes_in_binary)
#     if len(counts) == len(class_mean_code):
#         return class_mean_code
#     for i in range(len(codes_in_binary)):
#         code = codes_in_binary[i]
#         if counts[code] > 1:
#             while True:
#                 new_code = modify_code(code)
#                 if new_code not in codes_in_binary:
#                     codes_in_binary[i] = new_code
#                     counts[code] -= 1
#                     break
#
#     processed_class_mean_code = torch.zeros_like(class_mean_code)
#     for c, code in enumerate(codes_in_binary):
#         for i, b in enumerate(code):
#             processed_class_mean_code[c][i] = eval(b)
#     return processed_class_mean_code
#
#
# def get_class_mean_random(dataset, num_features, mean_range):
#     class_mean_code = torch.zeros((dataset.num_classes, num_features))
#     for c in range(dataset.num_classes):
#         random_order = torch.randperm(mean_range[0])
#         random_idx = random_order[:mean_range[1]]
#         class_mean_code[c][random_idx] = 1
#     processed_class_mean_code = process_class_mean_code(class_mean_code)
#     # torch.save(processed_class_mean_code, 'cifar10_class_mean1.pk')
#     return processed_class_mean_code
# def get_class_code(features=None, dataset=None, targets=None, class_mean_code=None, encode_type=None,
#                    based_important_feature=True, weight=None):
#     if class_mean_code is None:
#         if based_important_feature:
#             class_mean_code = torch.where(weight > 0, 1.0, 0.0)
#             # for i in range(features.shape[1]):
#             #     feature = features[:, i]
#             #     quantile = torch.quantile(feature, 0.75)
#             #     feature_min = torch.min(feature)
#             #     feature_max = torch.max(feature)
#             #     thresholds = (feature_min + feature_max) / 2.0
#             #     mean_feature = torch.mean(feature)
#             #     median_feature = torch.median(feature)
#             #     idx = targets[feature > thresholds]
#             #     idx = list(set(idx))
#             #     class_mean_code[idx] = 1
#         else:
#             features = torch.vstack(features)
#             targets = torch.hstack(targets)
#             targets = targets.cpu().numpy()
#             class_mean_code = torch.zeros((dataset.num_classes, features.shape[1]))
#             for idx, feature in enumerate(features):
#                 target = targets[idx]
#                 sigmoid_feature, encoded_feature = get_encoded_feature(feature, encode_type=encode_type)
#                 class_mean_code[target] += encoded_feature
#             nums = torch.tensor(dataset.train_num_per_cls_dict)
#             nums = torch.unsqueeze(nums, dim=1)
#             class_mean_code = torch.div(class_mean_code, nums)
#
#             _, class_mean_code = get_encoded_feature(class_mean_code, encode_type=encode_type)
#     processed_class_mean_code = process_class_mean_code(class_mean_code)
#     # torch.save(processed_class_mean_code, 'bs_cifar10_0.01_class_mean.pk')
#     return processed_class_mean_code


def get_features(dataloader=None, learning_model=None, device=None, config=None,
                 file=None, gpu_rank=0, train_model=False):
    features = []
    targets = []

    if check_file(file):
        features, targets = torch.load(file, weights_only=True)
    else:
        if train_model:
            learning_model.train()
        else:
            learning_model.eval()

        for i, (images, target) in enumerate(dataloader):
            images = images.to(device)
            output = learning_model(images, target)
            features.append(output['feature'].detach())
            targets.append(target)
            if i % config.config['trainer']['print_freq'] == 0:
                config.logger.info(f"batch {i}/{len(dataloader)} completed", gpu_rank)
        torch.save([features, targets], file)

    return features, targets


def check_common_features(dataset, learning_model, logger,
                          features=None, targets=None, num_per_cls=None, display=False,
                          ddp_on=False, gpu_rank=0):
    node = gpu_rank if display else 1
    if ddp_on:
        weights = learning_model.module.fc.weight.data
    else:
        weights = learning_model.fc.weight.data
    features = np.vstack(features)
    logger.info(f"feature shape: {features.shape}", gpu_rank)
    feature_size = features.shape[1]
    targets = np.hstack(targets)
    bin_num = 2
    dominant_class = np.ones((feature_size, bin_num), dtype=int) * (-1)
    dominant_class_percentage = np.ones((feature_size, bin_num), dtype=int) * (-1)
    dominant_class_bin_ratio = np.ones((feature_size, bin_num), dtype=int) * (-1)
    max_weight_class = np.ones(feature_size, dtype=int) * (-1)
    num_diff = 0
    num_same = 0
    max_weight_idx_num = np.ones(feature_size, dtype=int) * (-1)
    max_weight_idx_percentage = np.ones(feature_size, dtype=int) * (-1)
    max_weight_idx_ratio = np.ones(feature_size, dtype=int) * (-1)

    for feature_id in range(0, feature_size):
        # plot_feature_hist(weights[:, feature_id])
        logger.info(f"======Feature {feature_id}==================", node)

        weight_feature_i = weights[:, feature_id].detach().cpu().numpy()
        max_weight_class[feature_id] = np.argmax(weight_feature_i)
        logger.info(f"objects per class: {num_per_cls}", node)
        logger.info(f"weights ({weight_feature_i})", node)
        feature_i = features[:, feature_id]
        feature_min = min(feature_i)
        feature_max = max(feature_i)
        bin_width = 1e-6 + (feature_max - feature_min) / bin_num
        sorted_ids = np.argsort(feature_i)

        ub = bin_width + feature_min
        bin_id = 0

        feature_bins = [dict()]

        for i in sorted_ids:
            while True:
                if feature_i[i] <= ub:
                    if targets[i] not in feature_bins[bin_id]:
                        feature_bins[bin_id][targets[i]] = []
                    feature_bins[bin_id][targets[i]].append(feature_i[i])
                    break
                else:
                    if len(feature_bins[bin_id]) > 0:
                        _process_bin(bin_id, feature_bins, feature_id, dominant_class, dominant_class_percentage,
                                     dominant_class_bin_ratio,
                                     num_per_cls, None, logger, node)
                    feature_bins.append(dict())

                    bin_id += 1
                    ub = bin_width * (bin_id + 1) + feature_min
        assert bin_id == 1
        num_target_feature = dict()
        for t in feature_bins[0]:
            num_target_feature[t] = len(feature_bins[0][t])
        for t in feature_bins[1]:
            if t in num_target_feature:
                num_target_feature[t] += len(feature_bins[1][t])
            else:
                num_target_feature[t] = len(feature_bins[1][t])
        _process_bin(bin_id, feature_bins, feature_id, dominant_class, dominant_class_percentage,
                     dominant_class_bin_ratio,
                     num_per_cls, num_target_feature, logger, node)
        # logger.info(f"max-weight-class: {max_weight_class[feature_id]}")

        try:
            sorted_classes = sorted(feature_bins[bin_id].keys(), key=lambda t: len(feature_bins[bin_id][t]),
                                    reverse=True)
            max_weight_idx_num[feature_id] = sorted_classes.index(max_weight_class[feature_id])
        except ValueError:
            max_weight_idx_num[feature_id] = -1

        try:
            sorted_classes = sorted(feature_bins[bin_id].keys(), key=lambda t:
            (1.0 * len(feature_bins[bin_id][t])) / num_per_cls[t], reverse=True)
            max_weight_idx_percentage[feature_id] = sorted_classes.index(max_weight_class[feature_id])
        except ValueError:
            max_weight_idx_percentage[feature_id] = -1

        try:
            sorted_classes = sorted(feature_bins[bin_id].keys(), key=lambda t:
            (1.0 * len(feature_bins[bin_id][t])) / (num_target_feature[t]), reverse=True)
            max_weight_idx_ratio[feature_id] = sorted_classes.index(max_weight_class[feature_id])
        except ValueError:
            max_weight_idx_ratio[feature_id] = -1

        if dominant_class[feature_id, 1] != dominant_class_percentage[feature_id, 1]:
            # logger.info("Dominant classes are different.")
            num_diff += 1
        else:
            num_same += 1
    # logger.info(f"dominant class regarding number: {dominant_class}")
    # logger.info(f"dominant class regarding percentage: {dominant_class_percentage}")
    # plot_feature_hist(feature_i, bins=np.arange(feature_min, feature_max+bin_width, bin_width))
    logger.info("==========Summary=============", gpu_rank)
    logger.info(f"max_weight_idx_based_sorted_num: {max_weight_idx_num}", gpu_rank)
    logger.info(Counter(max_weight_idx_num), gpu_rank)
    logger.info(f"max_weight_idx_based_sorted_percentage: {max_weight_idx_percentage}", gpu_rank)
    logger.info(Counter(max_weight_idx_percentage), gpu_rank)
    logger.info(f"max_weight_idx_based_sorted_ratio: {max_weight_idx_ratio}", gpu_rank)
    logger.info(Counter(max_weight_idx_ratio), gpu_rank)
    for b in range(1, bin_num):
        logger.info(f"Bin {b}", gpu_rank)
        logger.info(f"dominant regarding feature num: {dominant_class[:, b]}", gpu_rank)
        stat_num = Counter(dominant_class[:, b])
        logger.info(stat_num, node)
        logger.info(f"dominant regarding feature percentage: {dominant_class_percentage[:, b]}", gpu_rank)
        stat_percentage = Counter(dominant_class_percentage[:, b])
        logger.info(stat_percentage, node)

        logger.info(f"dominant regarding feature ratio: {dominant_class_bin_ratio[:, b]}", gpu_rank)
        stat_ratio = Counter(dominant_class_bin_ratio[:, b])
        logger.info(stat_ratio, node)

        report = get_dominated_count(stat_num, dataset)
        logger.info(f"mean_num_of_dominant_feature_num: {report}", gpu_rank)

        report = get_dominated_count(stat_percentage, dataset)
        logger.info(f"mean_num_of_dominant_feature_percentage: {report}", gpu_rank)

        report = get_dominated_count(stat_ratio, dataset)
        logger.info(f"mean_num_of_dominant_feature_ratio: {report}", gpu_rank)

        max_weight_class_count = Counter(max_weight_class)
        report = get_dominated_count(max_weight_class_count, dataset)
        logger.info(f"mean_num_of_dominant_weight: {report}", gpu_rank)

    # logger.info(f"max_num!=max_percentage: {num_diff}, max_num=max_percentage: {num_same}", gpu_rank)


def _process_bin(bin_id, feature_bins, feature_id, dominant_class, dominant_class_percentage, dominant_class_bin_ratio,
                 num_per_cls, num_target_feature, logger,
                 node):
    values = list(feature_bins[bin_id].values())
    all_values = np.hstack(values)
    info = f"bin {bin_id}: max ({max(all_values)}) min ({min(all_values)}) num_objects"
    for t in feature_bins[bin_id].keys():
        info += f" [{t}: {len(feature_bins[bin_id][t])}]"
    info += " num_obj_%"
    for t in feature_bins[bin_id].keys():
        info += f" [{t}: {(1.0 * len(feature_bins[bin_id][t])) / num_per_cls[t]:.5f}]"

    logger.info(info, node)
    dominant_class[feature_id, bin_id] = max(
        feature_bins[bin_id].keys(),
        key=lambda target: len(feature_bins[bin_id][target]))
    dominant_class_percentage[feature_id, bin_id] = max(
        feature_bins[bin_id].keys(),
        key=lambda target: (1.0 * len(feature_bins[bin_id][target])) / num_per_cls[target]
    )
    if num_target_feature is not None:
        dominant_class_bin_ratio[feature_id, bin_id] = max(
            feature_bins[bin_id].keys(),
            key=lambda target:
            (1.0 * len(feature_bins[bin_id][target])) / (num_target_feature[target])
        )

    logger.info(f"dominant-class: {dominant_class[feature_id, bin_id]}, "
                f"dominant-class-percentage: {dominant_class_percentage[feature_id, bin_id]}, "
                f"dominant-class-ratio: {dominant_class_bin_ratio[feature_id, bin_id]}, ", node)


def _classify_one_sample(x_per, transform, device, learning_model):
    if isinstance(transform, torchvision.transforms.Compose):
        if len(transform.transforms) > 2:
            simply_transform = transform.transforms[0]
            input_image = simply_transform(x_per)
            rest_transforms = transform.transforms[1:]
        else:
            rest_transforms = transform.transforms
            input_image = x_per
        X = input_image
        for transform in rest_transforms:
            X = transform(X)
    else:
        input_image = None
        X = x_per

    X = X.unsqueeze(0)
    X = X.to(device)

    output = learning_model(X)
    return output, input_image


def get_all_modules(module, module_list):
    children = list(module.children())
    if len(children):
        for child in children:
            get_all_modules(child, module_list)
    else:
        module_list.append(module)
    return module_list


def normalize(matrix, p=2, dim=1, power=1):
    from utils.utils import check_nan_array_torch
    matrix = check_nan_array_torch(matrix)
    norm = torch.norm(matrix, p=p, dim=dim, keepdim=True)
    norm = check_nan_array_torch(norm)
    matrix_normalized = matrix / torch.pow(norm, power)
    return matrix_normalized


def adjust_loss(loss, logger, threshold=10.0):
    if loss <= threshold:
        return loss
    else:
        scalar = float(loss / threshold)
        new_loss = loss / scalar
        logger.info(f'Loss is adjusted: {loss} => {new_loss}')
    return new_loss


class NormedLinear(nn.Module):

    def __init__(self, in_features, out_features, s=30):
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(in_features, out_features))
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)
        self.s = s

    def forward(self, x):
        out = F.normalize(x, dim=1).mm(F.normalize(self.weight, dim=0))
        return out * self.s


def norm_linear_weight_grad(feature, g, weights):
    s = 30
    eps = 1e-12  # Try 1e-5 if this doesn't match your layer's epsilon
    # feature in shape of (D, C), g is the grad of logit
    # 4. Your Manual Calculation
    x_hat = F.normalize(feature, dim=1, eps=eps)
    w_hat = F.normalize(weights, dim=0, eps=eps)
    # g is the external_grad
    dL_dsim = g * s
    w_norm = torch.norm(weights, p=2, dim=0, keepdim=True)

    term1 = torch.mm(x_hat.T, dL_dsim)
    # dot = torch.sum(dL_dsim * sim, dim=0, keepdim=True) # Logic A
    dot = torch.sum(term1 * w_hat, dim=0, keepdim=True)  # Logic B (Equivalent)
    term2 = dot * w_hat

    manual_w_grad = (term1 - term2) / (w_norm + eps)
    return manual_w_grad


def ce_grad_func(ce_logits: torch.Tensor, ce_targets: torch.Tensor, num_classes: int, check: bool = True):
    probs = torch.softmax(ce_logits, dim=1)
    num = len(probs)
    # FIX 1: Ensure targets are on the correct device and cast to float
    # We create the one_hot, then cast to the same dtype as probs (e.g., float16/32)
    y = torch.nn.functional.one_hot(ce_targets, num_classes).to(dtype=probs.dtype)

    # Note: If you use 'ignore_index' (like -100), you must filter ce_targets before one_hot
    # or zero out the gradients for those rows here.

    logit_reward_grad = ((probs - 1) * y) / num  # Target class gets (p - 1)
    logit_penalty_grad = (probs * (1 - y)) / num  # Non-targets get (p)

    # Logic is correct: (p - 1) + p = 2p - 1? NO.
    # At target index k=y: (p-1)*1 + p*0 = p-1. Correct.
    # At non-target k!=y: (p-1)*0 + p*1 = p. Correct.
    logit_grad = logit_reward_grad + logit_penalty_grad

    details = (logit_reward_grad, logit_penalty_grad)
    if check:
        check_identical(logit_grad, ce_logits.grad, "Logit ")
    return logit_grad, details
