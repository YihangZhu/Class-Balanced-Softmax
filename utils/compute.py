from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms
from torchvision import transforms

from utils.randaugment import rand_augment_transform
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


def augmentation_randncls_func(size, ra_params, normalize, randaug_n=2, randaug_m=10):
    return [
        transforms.RandomResizedCrop(size, scale=(0.08, 1.)),
        transforms.RandomHorizontalFlip(),
        rand_augment_transform('rand-n{}-m{}-mstd0.5'.format(randaug_n, randaug_m), ra_params),
        transforms.ToTensor(),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.0)
        ], p=1.0),
        normalize,
    ]


def augmentation_randnclsstack_func(size, ra_params, normalize, randaug_n=2, randaug_m=10):
    return [
        transforms.RandomResizedCrop(size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomGrayscale(p=0.2),
        rand_augment_transform('rand-n{}-m{}-mstd0.5'.format(randaug_n, randaug_m), ra_params),
        transforms.ToTensor(),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        ], p=0.8),
        normalize,
    ]


def augmentation_sim_func(size, normalize):
    return [transforms.RandomResizedCrop(size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            ], p=0.8),
            normalize
            ]
