import numpy as np
import torch
import torch.nn.functional as F

from utils.utils import get_attribute, reduce_all

precision = 5


def get_negligible_count(weights, dataset, threshold: float):
    max_per_feature = torch.max(weights, dim=0)[0]
    min_per_feature = torch.min(weights, dim=0)[0]

    limits = min_per_feature + threshold * (max_per_feature - min_per_feature)
    mask = (weights <= limits) * 1.0
    h_m_t = torch.sum(mask, dim=1)

    report, group_mean = get_group_mean(dataset, h_m_t)
    return report, group_mean, h_m_t


def get_dominated_count(weights, dataset, threshold: float):
    max_per_feature = torch.max(weights, dim=0)[0]
    min_per_feature = torch.min(weights, dim=0)[0]

    limits = max_per_feature - threshold * (max_per_feature - min_per_feature)
    mask = (weights >= limits) * 1.0
    h_m_t = torch.sum(mask, dim=1)

    # max_weight_class = torch.argmax(weights, dim=0).tolist()
    # max_weight_class_count = Counter(max_weight_class)
    # h_m_t = torch.empty(dataset.num_classes, dtype=torch.float)
    # for c in range(dataset.num_classes):
    #     if c not in max_weight_class_count.keys():
    #         h_m_t[c] = 0
    #     else:
    #         h_m_t[c] = max_weight_class_count[c]

    report, group_mean = get_group_mean(dataset, h_m_t)
    return report, group_mean, h_m_t


def get_group_mean(dataset, values, sub_prefix='', measure='mean'):
    report = ''
    head, med, tail = None, None, None
    if isinstance(values, torch.Tensor):
        func = getattr(torch, measure)
    else:
        func = getattr(np, measure)
    if dataset.head_class_idx is not None:
        # temp_h = values[dataset.head_class_idx].detach().numpy()
        head = func(values[dataset.head_class_idx])
        report += f"{sub_prefix}@head: {head:.{precision}f}\t"
        if isinstance(head, torch.Tensor):
            head = head.detach().cpu().item()
    if dataset.med_class_idx is not None:
        # temp_m = values[dataset.med_class_idx].detach().numpy()
        med = func(values[dataset.med_class_idx])
        report += f"{sub_prefix}@med: {med:.{precision}f}\t"
        if isinstance(med, torch.Tensor):
            med = med.detach().cpu().item()
    if dataset.tail_class_idx is not None:
        # temp_t = values[dataset.tail_class_idx].detach().numpy()
        tail = func(values[dataset.tail_class_idx])
        report += f"{sub_prefix}@tail: {tail:.{precision}f}\t"
        if isinstance(tail, torch.Tensor):
            tail = tail.detach().cpu().item()

    return report, (head, med, tail)


def check_representation_model(saved_values, dataset, logger, rate, gpu_rank=None):
    if 'features' in saved_values:
        feature = saved_values['features']
        processed_values = torch.zeros((dataset.num_classes, len(feature[0][0])))
        for c in range(dataset.num_classes):
            class_mean = torch.tensor(np.array(feature[c])).mean(0)
            processed_values[c] = class_mean
        report, _ = check_statistic_info(processed_values, dataset, rate=rate)
        logger.info(f"Representation:\t{report}", gpu_rank)


def check_class_mean(learning_model, dataset, logger, rate, ddp_on, gpu_rank=None):
    classifier = get_attribute(learning_model, 'classifier', ddp_on=ddp_on)
    if hasattr(classifier, 'class_mean'):
        class_mean = classifier.class_mean
        report, _ = check_statistic_info(class_mean, dataset, rate=rate)
        logger.info(f"Class mean:\t{report}", gpu_rank)


def check_statistic_info(weights, dataset, rate):
    weight_sum = torch.sum(weights, dim=1, keepdim=True)
    weight_sum_report, weight_sum_mean = get_group_mean(dataset, weight_sum)

    weight_dominant_report, weight_dominant_mean, _ = get_dominated_count(weights, dataset, rate)

    weight_negligible_report, weight_negligible_mean, _ = get_negligible_count(weights, dataset, rate)

    norm = torch.norm(weights, p=2, dim=1, keepdim=True)
    norm_report, norm_mean = get_group_mean(dataset, norm)

    report = (f"weight-sum mean: {weight_sum_report}"
              f"\t#dominant-feature mean: {weight_dominant_report}"
              f"\t#negligible-feature mean: {weight_negligible_report}"
              f"\tweight-norm mean: {norm_report}")

    results = {'weights_head_med_tail': weight_sum_mean,
               'dominant-feature_head_med_tail': weight_dominant_mean,
               'negligible-feature_head_med_tail': weight_negligible_mean,
               'norm_head_med_tail': norm_mean
               }

    return report, results


def check_linear_classifier(learning_model, dataset, logger, ddp_on, gpu_rank, rate):
    classifier = get_attribute(learning_model, 'classifier', ddp_on)
    report = "fc:\t"
    if classifier is not None:
        fc = get_attribute(classifier, 'fc', ddp_on=False)
        if fc is not None:
            weights = fc.weight.detach()
            if weights.shape[0] != dataset.num_classes:
                weights = torch.permute(weights, (1, 0))

            weight_report, results = check_statistic_info(weights, dataset, rate=rate)
            report += weight_report
            bias = get_attribute(fc, 'bias', ddp_on=False)
            if bias is not None:
                bias_mean_report, bias_mean = get_group_mean(dataset, bias.detach())
                report += f"\tbias mean: {bias_mean_report}"
                results["bias_head_med_tail"] = bias_mean

            logger.info(report, gpu_rank=gpu_rank)
            return results
    return None


class TrainRecorder:
    def __init__(self, keys):
        self._saved_values = dict()
        self._saved_values['last_train_details'] = None
        self._saved_values['train_recall'] = []
        self._saved_values['train_precision'] = []
        self._saved_values['train_f1_score'] = []
        self._saved_values['loss'] = dict()
        self._saved_values['loss']['train'] = []
        if keys is not None:
            self._saved_values['last_eval_details'] = dict()
            self._saved_values['eval_recall'] = dict()
            self._saved_values['eval_precision'] = dict()
            self._saved_values['eval_f1_score'] = dict()
            for key in keys:
                self._saved_values['last_eval_details'][key] = None
                self._saved_values['eval_recall'][key] = []
                self._saved_values['eval_precision'][key] = []
                self._saved_values['eval_f1_score'][key] = []
                self._saved_values['loss']['test_' + key] = []

        self._saved_values['weights_head_med_tail'] = []
        self._saved_values['dominant-feature_head_med_tail'] = []
        self._saved_values['negligible-feature_head_med_tail'] = []
        self._saved_values['bias_head_med_tail'] = []
        self._saved_values['norm_head_med_tail'] = []

    def update_all_record(self, saved_values, logger, rank):
        for key in saved_values.keys():
            if key in self._saved_values:
                self._saved_values[key] = saved_values[key]
        logger.info(self.print_train_last(), rank)
        if 'eval_recall' in self._saved_values:
            for key in self._saved_values['eval_recall'].keys():
                logger.info(self.print_eval_last(key=key), rank)

    def update_train_record(self, t_report, metrics_values, loss_values):
        if metrics_values is not None:
            self._saved_values['train_recall'].append(metrics_values['recall'])
            self._saved_values['train_precision'].append(metrics_values['precision'])
            self._saved_values['train_f1_score'].append(metrics_values['f1_score'])
            self._saved_values['last_train_details'] = t_report

        for name, value in loss_values.items():
            if name != '':
                name = f'train_{name}'
            else:
                name = 'train'
            if name not in self._saved_values['loss']:
                self._saved_values['loss'][name] = []
            self._saved_values['loss'][name].append(value)

    def update_eval_record(self, v_report, metrics_values, key, loss_values):
        self._saved_values['eval_recall'][key].append(metrics_values['recall'])
        self._saved_values['eval_precision'][key].append(metrics_values['precision'])
        self._saved_values['eval_f1_score'][key].append(metrics_values['f1_score'])
        if loss_values is not None:
            for name, value in loss_values.items():
                if name != '':
                    name = f'test_{key}_{name}'
                else:
                    name = f'test_{key}'
                if name not in self._saved_values['loss']:
                    self._saved_values['loss'][name] = []
                self._saved_values['loss'][name].append(value)
        self._saved_values['last_eval_details'][key] = v_report

    def print_train_last(self, prefix='Train'):
        """training accuracy may be slightly different between different GPUs"""
        return f"* {prefix}: {self._saved_values['last_train_details']}"

    def print_eval_last(self, key, prefix='Eval'):
        return f"* {key} {prefix}: {self._saved_values['last_eval_details'][key]}"

    def update_linear_classifier_record(self, dict_):
        for key, value in dict_.items():
            self._saved_values[key].append(value)

    def _clean(self, saved_value: dict):
        key_list = list(saved_value.keys())
        for key in key_list:
            record = saved_value[key]
            if isinstance(record, dict):
                self._clean(record)
            elif isinstance(record, list):
                if len(record) == 0 or record[0] is None:
                    del saved_value[key]

    def get_saved_values(self, clean=False):
        if clean:
            self._clean(self._saved_values)
        return self._saved_values


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

    def __repr__(self):
        return self.__str__()


def _get_batch_fmtstr(num_batches):
    num_digits = len(str(num_batches // 1))
    fmt = '{:' + str(num_digits) + 'd}'
    return '[' + fmt + '/' + fmt.format(num_batches) + ']'


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix="", print_freq=None):
        self.batch_fmtstr = _get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix
        self.print_freq = print_freq

    def display(self, batch, logger, gpu_id, add=None):
        if self.print_freq is not None:
            do_print = batch % self.print_freq == 0
        else:
            do_print = True
        if do_print:
            entries = [self.prefix + self.batch_fmtstr.format(batch)]
            entries += [str(meter) for meter in self.meters]
            if add is not None:
                entries.append(add)
            logger.info('\t'.join(entries), gpu_rank=gpu_id)


class ImbalanceAccuracy:
    def __init__(self, dataset, device):
        self.num_classes = dataset.num_classes
        self.dataset = dataset
        # total number of predicated correctly objects in each class
        self.true_pos = torch.zeros(self.num_classes).to(device)
        self.false_pos = torch.zeros(self.num_classes).to(device)
        self.false_neg = torch.zeros(self.num_classes).to(device)
        # true positive + false negative equals total number of objects in each class regarding the instances loaded

    def update(self, target, output=None, predicted=None):
        if predicted is None:
            _, predicted = output.max(1)  # return the index of the maximum value at dimension 1.
        target_one_hot = F.one_hot(target, self.num_classes)  # B x num_class matrix.
        predict_one_hot = F.one_hot(predicted, self.num_classes)  # B x num_class matrix

        self.true_pos += (target_one_hot + predict_one_hot == 2).sum(dim=0).to(torch.float)
        self.false_pos += (predict_one_hot - target_one_hot == 1).sum(dim=0).to(torch.float)
        self.false_neg += (predict_one_hot - target_one_hot == -1).sum(dim=0).to(torch.float)

    def calculate(self, last_epoch, use_ddp, logger):
        if last_epoch:
            if use_ddp:
                self.true_pos = reduce_all(self.true_pos, 'SUM', logger)
                self.false_pos = reduce_all(self.false_pos, 'SUM', logger)
                self.false_neg = reduce_all(self.false_neg, 'SUM', logger)

        precision_classes = self.true_pos / (self.true_pos + self.false_pos)
        recall_classes = self.true_pos / (self.true_pos + self.false_neg)
        precision_classes = precision_classes.detach()
        recall_classes = recall_classes.detach()

        report, recall_values = self._calculate(recall_classes, 'acc', "[recall] ")
        group_report, precision_values = self._calculate(precision_classes, 'pcs', "[precision] ")
        report += group_report

        F1_score = 2 * ((precision_classes * recall_classes) / (precision_classes + recall_classes + 1e-12))
        group_report, F1_score_values = self._calculate(F1_score, 'f1', "[f1_score] ")
        report += group_report

        return (report, {'recall': recall_values, 'precision': precision_values, 'f1_score': F1_score_values},
                {'recall': recall_classes, 'precision': precision_classes, 'f1_score': F1_score})

    def _calculate(self, metric_values, sub_prefix, prefix=''):
        metric_values = check_nan(metric_values) * 100
        assert hasattr(self.dataset, 'head_class_idx')
        group_report, group_values = get_group_mean(self.dataset, metric_values, sub_prefix=sub_prefix)
        report = f'{prefix} {group_report}'
        ave_acc_classes = metric_values.mean().item()  # average accuracy across all the classes. 1/C\sum_{i}n(i,i)/N_i
        sd_acc_classes = metric_values.std().item()
        report += f"{sub_prefix}@class: {ave_acc_classes:.{precision}f}\t{sub_prefix}@sd: {sd_acc_classes:.{precision}f}\t"
        if len(metric_values) <= 10:
            report += f"{sub_prefix}@class{[round(a, 3) for a in metric_values.detach().cpu().numpy()]}\t"

        return report, tuple([i for i in group_values] + [ave_acc_classes])


def check_nan(tensors):
    if torch.any(torch.isnan(tensors)):
        tensors = torch.nan_to_num(tensors)
    return tensors


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            # https://discuss.pytorch.org/t/when-and-why-do-we-use-contiguous/47588
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res, pred


def iou_pytorch(outputs: torch.Tensor, labels: torch.Tensor):
    epsilon = 1e-6
    # You can comment out this line if you are passing tensors of equal shape
    # But if you are passing output from UNet or something it will most probably
    # be with the BATCH x 1 x H x W shape
    intersection = ((outputs * labels) > 0).sum((1, 2))  # Will be zero if Truth=0 or Prediction=0
    union = ((outputs + labels) > 0).sum((1, 2))  # Will be zzero if both are 0
    iou = (intersection + epsilon) / (union + epsilon)  # We smooth our devision to avoid 0/0
    return iou
