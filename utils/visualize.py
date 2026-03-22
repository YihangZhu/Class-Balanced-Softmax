import time

import matplotlib.pyplot as plt
from scipy import stats

# EXAMPLE: For a standard double-column column (approx 3.3 inches)
# If your paper is single-column, this is usually approx 6.5 inches
TARGET_WIDTH_INCHES = 6.5
length_width_rate = 0.45
# length_width_rate = 1.3
plt.rcParams.update({
    "figure.figsize": (TARGET_WIDTH_INCHES, TARGET_WIDTH_INCHES * length_width_rate),  # Fixed width
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Times"],
    "font.size": 10,  # Matches your LaTeX [10pt]{article}
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "xtick.labelsize": 8,  # Usually 1-2pts smaller looks more professional
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "text.latex.preamble": (
        r"\usepackage{amsmath} "
        r"\usepackage{amsfonts} "
        r"\DeclareMathAlphabet{\mathcal}{OMS}{cmsy}{m}{n}"
    )
})


def save_fig(fig_name):
    plt.savefig(fig_name, bbox_inches='tight', pad_inches=0.01)


from utils.utils import unpickle
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from utils.compute import _classify_one_sample
from utils.meter import get_group_mean
from datasets.checkpoints import get_checkpoint
from utils.utils import kl_divergence


def plot_train_loss(saved_values, file):
    if 'loss' in saved_values:
        assert type(saved_values['loss']) is dict
        fig, ax = plt.subplots()
        colors = [
            'blue', 'red', 'green', 'orange', 'olive', 'purple', 'gray',
            'brown', 'pink', 'cyan', 'magenta', 'gold', 'teal', 'navy',
            'coral', 'lime', 'indigo', 'maroon', 'turquoise', 'darkgreen'
        ]
        for i, (key, loss) in enumerate(saved_values['loss'].items()):
            if len(loss) == 0:
                continue
            ax.plot(loss, color=colors[i], label=f'{key.replace("_", "-")}')
        ax.set(xlabel='epoch',
               ylabel='loss')
        ax.legend()
        save_fig(file)
        file = str(file).replace('loss.pdf', 'loss_brief.pdf')
        ax.set_ylim(top=5)
        save_fig(file)
        plt.close()


keys = ['norm_head_med_tail', 'weights_head_med_tail', 'bias_head_med_tail', 'dominant-feature_head_med_tail',
        'negligible-feature_head_med_tail',
        'train_recall', 'train_precision', 'train_f1_score', 'eval_recall', 'eval_precision', 'eval_f1_score']
info = {
    'norm_head_med_tail': {
        'file': 'weight_norm.pdf',
        'ylabel': 'Weight L2-norm average over classes'
    },
    'weights_head_med_tail': {
        'file': 'weight_sum.pdf',
        'ylabel': 'Weight sum average over classes'
    },
    'bias_head_med_tail': {
        'file': 'bias.pdf',
        'ylabel': 'Bias average over classes'
    },
    'dominant-feature_head_med_tail': {
        'file': 'feature_dominant.pdf',
        'ylabel': 'Num feature dominated per class'
    },
    'negligible-feature_head_med_tail': {
        'file': 'feature_negligible.pdf',
        'ylabel': 'Num negligible feature per class'
    },
    'train_recall': {
        'file': 'train_recall_per_group.pdf',
        'ylabel': 'Recall per group (%) in training'
    },
    'train_precision': {
        'file': 'train_precision_per_group.pdf',
        'ylabel': 'Precision per group (%) in training'
    },
    'train_f1_score': {
        'file': 'train_f1_score_per_group.pdf',
        'ylabel': 'F1-score per group (%) in training'
    },
    'eval_recall': {
        'file': 'test_recall_per_group.pdf',
        'ylabel': 'Recall per group (%) in testing'
    },
    "eval_precision": {
        'file': 'test_precision_per_group.pdf',
        'ylabel': 'Precision per group (%) in testing'
    },
    'eval_f1_score': {
        'file': 'test_f1_score_per_group.pdf',
        'ylabel': 'F1-score per group (%) in testing'
    },
}


def _plot_results(save_path, saved_values):
    plot_head_med_tail_values(saved_values, save_path)
    plot_train_test_metrics(['recall', 'precision', 'f1_score'], saved_values, save_path)
    plot_train_loss(saved_values, save_path / 'loss.pdf')
    plot_probs(saved_values, save_path / 'probs_class0.pdf')


def plot_results(saved_values, config):
    sns.set_theme()
    assert isinstance(saved_values, dict)
    for key, saved_value in saved_values.items():
        saved_path = config.dirs['save_path'] / f"plots_{key}"
        saved_path.mkdir(parents=True, exist_ok=True)
        _plot_results(saved_path, saved_value)


def plot_head_med_tail_values(head_med_tail_values, parent_path):
    def plot_values(_values, _name=None):
        _values = np.vstack(_values)
        head, med, tail = _values[:, 0], _values[:, 1], _values[:, 2]
        fig, ax = plt.subplots()
        ax.plot(head, color='blue', label='head classes')
        if None not in med:
            ax.plot(med, color='red', label='med classes')
        ax.plot(tail, color='green', label='tail classes')
        ax.set(xlabel='Epoch',
               ylabel=info[key]['ylabel'])
        ax.legend()
        file_name = info[key]['file']
        if _name is not None:
            file_name = _name + '_' + file_name
        save_fig(parent_path / file_name)
        plt.close()

    for key in keys:
        if key in head_med_tail_values:
            values = head_med_tail_values[key]
            if isinstance(values, dict):
                values_list = values
                for name, values in values_list.items():
                    plot_values(values, name)
            else:
                plot_values(values)


def plot_train_test_metrics(metrics, saved_values, file_dir):
    """
    Function for plotting training and validation loss
    """
    for metric in metrics:
        if f'train_{metric}' in saved_values:
            train_values = saved_values[f'train_{metric}']

            file = f'{file_dir}/train_test_{metric}.pdf'
            # temporarily change the style of the plots to seaborn
            fig, ax = plt.subplots()
            train_values = np.vstack(train_values)
            ax.plot(train_values[:, 3], color='blue', label='train')
            line_style = ['-', '-*', '->', '--']
            count = 0
            if f'eval_{metric}' in saved_values:
                eval_values = saved_values[f'eval_{metric}']
                for key, item in eval_values.items():
                    if len(item) == 0:
                        continue
                    eval_value = np.vstack(item)
                    ax.plot(eval_value[:, 3], line_style[count], color='red', label=f'test_{key}')
                    count += 1
            ax.set(title=f"Average {metric.replace('_', '-')} over classes at each epoch",
                   xlabel='Epoch',
                   ylabel=f"{metric.replace('_', '-')} (%)")
            ax.legend()
            save_fig(file)
            plt.close()


def _get_plot_title(similarities, true_class_name, topk, class_names, dataset=None):
    prob = F.softmax(similarities, dim=1)
    prob = prob[0]

    title = f'true:{true_class_name}'
    title = add_class_group(title, true_class_name, dataset)
    prob_topk, pred_topk = prob.topk(topk, 0, True, True)
    for k in range(topk):
        predict_class_name = class_names[pred_topk[k]]
        predict_prob = prob_topk[k] * 100
        title += f'\n{predict_class_name} ({predict_prob:.0f}%)'
        title = add_class_group(title, predict_class_name, dataset)
    return title


def add_class_group(title, predict_class_name, dataset):
    if dataset is not None:
        if predict_class_name in dataset.head_class_idx:
            title += ' h'
        elif predict_class_name in dataset.med_class_idx:
            title += 'm'
        elif predict_class_name in dataset.tail_class_idx:
            title += 't'
    return title


# data is a matrix with each row representing a sample
# targets is a column with each element representing the label


def plot_image(x_per):
    plt.figure()
    plt.imshow(x_per)
    plt.show()


def plot_probs(saved_values, name=None):
    if 'record_probs' in saved_values:
        record_probs = saved_values['record_probs']
        plt.figure()

        plt.ylabel('Total number of samples')
        plt.xlabel('Probability of a sample being class 0')
        plt.hist(record_probs)
        # v = np.round(np.mean(record_probs), 2)
        # plt.xticks([v])
        plt.xlim([0.3, 1])
        plt.tight_layout()
        if name is None:
            plt.show()
        else:
            save_fig(name)


def plot_classifier_parameters(file_name=None, values=None, values2=None, feature_id=''):
    if values is not None:
        plt.figure()
        plt.ylabel(f'Feature {feature_id} value')
        plt.xlabel('Sorted class IDs regarding number of training samplers per class')
        plt.plot(values, label='iNaturalist2018')
        if values2 is not None:
            plt.plot(values2, label='CIFAR100-LT100')
        # v = np.round(np.mean(record_probs), 2)
        # plt.xticks([v])
        # plt.xlim([0.3, 1])
        # plt.xticks([])
        plt.legend()
        plt.tight_layout()
        if file_name is None:
            plt.show()
        else:
            save_fig(file_name)
    return values


def plot_num_sample_per_cls(dataset, name=None):
    import scipy.stats as stats

    plt.figure()
    sorted_ids = dataset.sorted_id
    num_samples = (dataset.train_num_per_cls_dict)
    total = np.sum(num_samples)
    rate = num_samples / total
    _min_rate = np.array([1 / len(num_samples)] * len(num_samples))
    kl = kl_divergence(_min_rate, rate)
    skewness = stats.skew(rate)
    kurtosis = stats.kurtosis(rate)
    print('skewness: {}'.format(skewness))
    print('kurtosis: {}'.format(kurtosis))
    print('kl: {}'.format(kl))
    exit()

    # avg = total/len(num_samples)

    # power law
    def power_law_p(beta):
        p = rate * np.power(num_samples, beta)
        return p / sum(p)

    power_law_w_ib = power_law_p(0.3)
    power_law_w_b = power_law_p(-0.8)

    # cos
    scalar = max(num_samples) / (np.pi / 2)

    def cos_p(beta):
        cos_b = rate * (np.cos(num_samples / scalar + 3 / 2 * np.pi)) ** (beta)
        cos_b /= sum(cos_b)
        return cos_b

    cos_ib = cos_p(0.3)
    cos_b = cos_p(-2)

    max_n = max(num_samples)
    min_n = min(num_samples)

    def log_p(beta):
        p = rate * np.log(1 + (num_samples - min_n + 1) / (max_n - min_n + 1) * (np.e - 1)) ** beta
        return p / sum(p)

    log_p_ib = log_p(0.1)
    log_p_b = log_p(-1)

    # # exponential
    # def exp_p(beta):
    #     p = rate * np.exp(beta*num_samples)
    #     return p/sum(p)
    # exp_p_ib = exp_p(0.1)
    # exp_p_b = exp_p(-0.1)

    plt.plot(rate[sorted_ids], label='num')
    plt.plot(power_law_w_ib[sorted_ids], label='power_ib')
    plt.plot(power_law_w_b[sorted_ids], label='power_b')
    # plt.plot(cos_ib[sorted_ids], label='cos_ib')
    # plt.plot(cos_b[sorted_ids], label='cos_b')
    # plt.plot(log_p_ib[sorted_ids], label='log_ib')
    # plt.plot(log_p_b[sorted_ids], label='log_b')
    # plt.plot(exp_p_ib[sorted_ids], label='exp_ib')
    # plt.plot(exp_p_b[sorted_ids], label='exp_b')

    plt.ylabel('Weights', fontsize=15)
    plt.xlabel('Sorted class IDs regarding number of training samplers per class', fontsize=15)
    plt.legend(fontsize=15)
    plt.tight_layout()
    if name is None:
        plt.show()
    else:
        plt.savefig(name)


def plot_feature_hist(feature_i_list, name=None, bins=None):
    plt.figure()

    plt.ylabel('Total number of classes')
    plt.xlabel('Value of the feature')
    plt.hist(feature_i_list, bins=bins)
    # v = np.round(np.mean(record_probs), 2)
    # plt.xticks([v])
    # plt.xlim([0.3, 1])
    plt.tight_layout()
    if name is None:
        plt.show()
    else:
        save_fig(name)
    plt.close()


def plot_heatmap(heatmap_arr, background=None, transparent=1, file_path=None, similarities=None,
                 target=None, topk=5, class_names=None, colourmap='jet', dataset=None):
    plt.figure()
    if isinstance(heatmap_arr, torch.Tensor):
        heatmap_arr = heatmap_arr.detach().cpu().numpy()
    if background is not None:
        if isinstance(background, np.ndarray):
            if len(background.shape) == 3 and background.shape[0] == 3:
                background = np.transpose(background, (1, 2, 0))
        plt.imshow(background)
    if len(heatmap_arr.shape) == 3:
        colourmap = None
    sc = plt.imshow(heatmap_arr, alpha=transparent, cmap=colourmap, interpolation='nearest')
    if colourmap is not None:
        plt.colorbar(sc)
    plt.ylabel("")
    if dataset is not None:
        title = _get_plot_title(similarities, target, topk, class_names, dataset)
        plt.title(title)

    plt.axis('off')
    plt.tight_layout()
    if file_path is None:
        plt.show()
    else:
        save_fig(file_path)
    plt.close()


def visualise_prob_rate(dataset, method):
    plt.figure()
    num_per_class = dataset.train_num_per_cls_dict
    average = np.mean(num_per_class, axis=0)
    sorted_ids = dataset.sorted_id

    file_path = get_checkpoint(data_name=dataset.key, method=method, _type='class_prob')

    practice_ib = unpickle(f"{file_path}/class_prob_train.pk")

    practice_b = unpickle(f"{file_path}/class_prob_b_train.pk")
    power = practice_ib / practice_b
    power = power.detach().numpy()
    # power = np.log(power) - 1.5*np.log(num_per_class)
    plt.plot(power[sorted_ids])
    # plt.hlines(1, 0, len(power), colors='red', linestyles='--')
    # plt.ylim((-5,5))
    plt.show()
    # print(torch.mean(power))


def visualise_class_prob(data_types, dataset, methods):
    plt.figure()
    file_name = f"{dataset.key}_train_b.pdf"
    # line_type = [(0, ()), (0, (5, 10)), (0, (1, 10))]
    num_per_class = dataset.train_num_per_cls_dict
    total_num = np.sum(num_per_class)

    for i, method in enumerate(methods):
        data_type = data_types[i]
        file_path = get_checkpoint(data_name=dataset.key, method=method, _type='class_prob')
        if data_type == 'train':
            file_path += "/class_prob_train.pk"
            expected_p = num_per_class / total_num
            plot_label = r"$\hat{p}(y=c)$"
        elif data_type == 'train_b':
            file_path += "/class_prob_b_train.pk"
            expected_p = num_per_class / total_num
            plot_label = r"$p(y=c)$"
        elif data_type == 'test_b':
            file_path += "/class_prob_b_eval.pk"
            expected_p = 1.0 / dataset.num_classes
            plot_label = "test"
        else:
            file_path += "/class_prob_eval.pk"
            expected_p = 1.0 / dataset.num_classes
            plot_label = "test"
        try:
            practice_p = torch.load(file_path, map_location=torch.device('cpu'), weights_only=True)
        except Exception as e:
            print(e)
            exit(-1)
        sorted_ids = dataset.sorted_id
        if isinstance(expected_p, float):
            expected_p = np.array([expected_p] * practice_p.shape[0])
        if isinstance(practice_p, torch.Tensor):
            practice_p = practice_p.detach().cpu().numpy()
        #
        kl = kl_divergence(expected_p, practice_p)
        diff = (practice_p - expected_p) / expected_p * 100
        # diff = practice_p
        report_mean, _ = get_group_mean(dataset, diff, measure='mean')
        report_var, _ = get_group_mean(dataset, diff, measure='std')
        print(f"{method} {dataset.key} {data_type} mean: {report_mean} std: {report_var}")

        plot_label = f"{plot_label}"

        print(f"{plot_label} kl divergence is {kl}")
        # if 'train' in data_type:
        #     plt.plot(diff[sorted_ids], label=plot_label)
        # else:
        plt.plot(practice_p[sorted_ids], "-", label=plot_label)
        # average = total_num / len(sorted_ids)
        # expected_p = expected_p * (num_per_class / average) ** (-1 / 2)
    plt.plot(expected_p[sorted_ids], "--", label=r"$\frac{|\mathcal{N}_{c}|}{|\mathcal{N}|}$")
    # practice_p = torch.load("saved/results_for_report/class_prob/res/2024-09-24_20-12-34_res50_imagenet_lt_0_gpus1/class_prob_b_train.pk", map_location=torch.device('cpu'), weights_only=True)
    # plt.plot(practice_p.detach().numpy()[sorted_ids], c='olive', linestyle=":",  label='imageNet-1K')
    # if 'train' in data_type:
    #     plt.plot(expected_p[sorted_ids], expected_p[sorted_ids], label="expected")
    # else:
    #     plt.hlines(1 / len(sorted_ids), 0, xmax=len(sorted_ids), colors='r', linestyles="-.",
    #                label=r"$\frac{1}{|\mathcal{C}|}$")
    # plt.hlines(0, 0, xmax=max(expected_p), colors='r', linestyles="-.",label="Expected")
    plt.ylabel(r'Class probability')
    # plt.ylim([-100, 100])
    plt.xlabel("Classes ranked by training set frequency (descending)")
    plt.xticks()
    plt.yticks()
    plt.legend()
    plt.tight_layout()
    if file_name:
        save_fig(file_name)
    plt.show()


def calculate_conflict(saved_values, dataset=None, method="", logger=None):
    def get_b_conflict():
        a = 100 * (saved_values['bias_grads_reward'] - saved_values['bias_grads_penalty']) / \
            saved_values['bias_grads_reward']
        # if isinstance(a, torch.Tensor):
        #     a = a.detach().cpu().numpy()
        return a

    def get_w_conflict():
        a = 100 * (saved_values['weight_grads_reward'] - saved_values['weight_grads_penalty']) / \
            saved_values['weight_grads_reward']
        # if isinstance(a, torch.Tensor):
        #     a = a.detach().cpu().numpy()
        return a.mean(axis=1)

    b_conflict = get_b_conflict()
    w_conflict = get_w_conflict()

    # plt.plot(b_conflict)
    # plt.title(f"b_conflict, epoch {epoch+1}, sum={_sum}")
    # plt.show()
    w_report_mean, _ = get_group_mean(dataset, w_conflict, measure='mean')
    b_report_mean, _ = get_group_mean(dataset, b_conflict, measure='mean')

    # w_conflict += max(-min(w_conflict), 0)
    # b_conflict += max(-min(b_conflict), 0)

    w_conflict = torch.softmax(w_conflict, dim=0)
    b_conflict = torch.softmax(b_conflict, dim=0)

    expected = torch.ones_like(w_conflict) / len(w_conflict)

    logger.info(f"{method} {dataset.key} weight: {w_report_mean} weight_kl: {kl_divergence(expected, w_conflict)}\t"
                f"bias: {b_report_mean} bias_kl: {kl_divergence(expected, b_conflict)}")


def check_conflict(dataset, method, logger=None):
    file_path = get_checkpoint(data_name=dataset.key, method=method, _type='conflict') + "/grads.pk"
    # name = "2025-05-07_13-56-51_imagenet_lt_cbs_p_batch128_1-1_gpus1"
    # file_path = f"saved/train/imagenet_lt_cbs_p/{name}/grads.pk"
    # print(name)
    saved_values = torch.load(file_path, map_location=torch.device('cpu'), weights_only=False)
    calculate_conflict(saved_values, dataset=dataset, method=method, logger=logger)


def model_preference_issue():
    values = unpickle(
        "saved/results_for_report/linear_layer/softmax/res_bias/result/2023-07-22_17-41-33_res50_imagenet_lt_0_gpus4/ckps/last_ckp.tar")
    train_recall = values["saved_values"]['train_recall']
    train_recall = np.vstack(train_recall)
    eval_recall = values["saved_values"]['eval_recall']['balanced']
    eval_recall = np.vstack(eval_recall)
    plt.figure()
    plt.plot(train_recall[:, 0], color='red', label='train_head')
    plt.plot(train_recall[:, 1], color='blue', label='train_medium')
    plt.plot(train_recall[:, 2], color='green', label='train_tail')

    plt.plot(eval_recall[:, 0], '--', label='test_head', color='red')
    plt.plot(eval_recall[:, 1], '--', label='test_medium', color='blue')
    plt.plot(eval_recall[:, 2], '--', label='test_tail', color='green')

    # plt.plot(train_recall[:, 3], label='train_all')
    # plt.plot(eval_recall[:, 3], label='eval_all')
    plt.ylabel(r'Training/testing recall ($\%$)')
    plt.xlabel('Epoch')
    plt.legend()
    save_fig("preference.pdf")
    plt.show()


def show_img(image, file_name=None, title=None):
    plt.figure()
    plt.imshow(image)
    plt.axis('off')
    if title is not None:
        plt.title(title)
    if file_name is not None:
        save_fig(file_name)
    else:
        plt.show()
    plt.close()

def check_probs_target_class(dataset):
    balanced_model = torch.load(
        'saved/results_for_report/prob_target_class/2026-03-06_17-59-25_imagenet_lt_gpus4_seed0/target_probs_0.pk',
        map_location='cpu')
    imbalanced_model = torch.load(
        'saved/results_for_report/prob_target_class/2026-03-06_18-06-30_imagenet_lt_gpus4_seed0/target_probs_0.pk',
        map_location='cpu')

    def get_probs(values):
        probs = values['probs']
        num_classes = dataset.num_classes
        probs_class = torch.zeros(num_classes)
        targets = values['targets']

        for i in range(len(probs_class)):
            masks = targets == i
            mean_prob = torch.mean(probs[masks])
            probs_class[i] = mean_prob
        return probs_class

    balanced_probs = get_probs(balanced_model)
    imbalanced_probs = get_probs(imbalanced_model)

    plt.figure()
    plt.plot(balanced_probs[dataset.sorted_id], label='Balanced')
    plt.plot(imbalanced_probs[dataset.sorted_id], label='Imbalanced')
    plt.ylabel(r'Mean of target class probabilities')
    plt.xlabel("Classes ranked by training set frequency (descending)")
    plt.legend()
    plt.tight_layout()
    save_fig("prob_target_class_imagenet-lt.pdf")
    plt.show()


def visualise_grad_balancing():
    # Data from the LaTeX table
    data = {
        'Dataset': [
            'C10', 'C10-LT10', 'C10-LT50', 'C10-LT100',
            'C100', 'C100-LT10', 'C100-LT50', 'C100-LT100',
            'ImageNet-1K', 'ImageNet-LT', 'Place-LT'
        ],
        'Softmax_Head': [-0.06, 94.11, 96.65, 97.27, -0.47, -13.98, -10.16, -12.03, 6.15, 36.84, 13.66],
        'Softmax_Medium': [0.61, 28.00, -0.69, -21.16, -1.68, -60.40, -173.90, -256.53, -13.85, -212.28, -274.52],
        'Softmax_Tail': [-2.94, -1811.39, -4012.50, -5490.95, -0.48, -66.31, -302.27, -527.87, 10.63, -993.80,
                         -2197.76],
        'BS_Head': [-0.06, 77.22, 69.03, 63.81, -0.47, -76.80, -93.21, -102.92, 6.53, 0.67, -19.80],
        'BS_Medium': [0.61, 6.70, -32.28, -52.58, -1.68, -7.43, -13.48, -15.73, -13.03, -21.37, -4.06],
        'BS_Tail': [-2.94, -759.68, -824.68, -840.70, -0.48, 35.61, 33.31, 29.59, 11.10, 0.82, -32.59],
        'CBS_Head': [-0.06, 56.83, 16.81, -10.98, -0.47, -83.16, -96.99, -104.62, 6.58, 0.18, 10.35],
        'CBS_Medium': [0.61, 4.20, -5.46, -2.27, -1.68, -1.15, 0.51, 1.98, -12.95, -2.15, 68.41],
        'CBS_Tail': [-2.94, -423.68, -248.40, -187.73, -0.48, 43.72, 48.69, 48.89, 11.14, 30.59, 81.68]
    }

    df = pd.DataFrame(data)

    # Melt the dataframe for easier plotting with seaborn
    melted_df = pd.melt(df, id_vars=['Dataset'], var_name='Metric', value_name='Value')
    melted_df[['Method', 'Group']] = melted_df['Metric'].str.split('_', expand=True)

    # Define a function to generate the plot
    sns.set_theme(style="whitegrid")

    # Figure 2: Heatmap for all datasets
    # Reshape for heatmap
    # We want rows: Dataset, columns: (Method, Group)
    heatmap_data = df.set_index('Dataset')

    # Rename columns to be more readable
    heatmap_data.columns = [c.replace('_', ' ') for c in heatmap_data.columns]

    # Since values vary wildly (from 100 to -5000), a linear heatmap won't work well.
    # We use a robust scaler or symlog to visualize.
    # However, for a journal, showing the signs is key.

    plt.figure()
    # We will use a diverging color map. Values near zero are "balanced".
    # Mask out extreme values or use a non-linear color scale.
    # Let's try to clip for visualization purposes or just use a symlog scale.

    # Clip extreme values for better color distribution in heatmap
    clipped_data = heatmap_data.clip(lower=-500, upper=100)

    ax = sns.heatmap(clipped_data, annot=heatmap_data.values,
                     fmt=".1f", cmap="RdBu", center=0,
                     annot_kws={"weight": "bold", "size": 10},
                     cbar_kws={'label': 'Clipped Gradient Balance Intensity'})
    # Customising the X-axis for grouped labels
    new_labels = ['Head', 'Medium', 'Tail'] * 3
    ax.set_xticklabels(new_labels, rotation=0)
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.tick_params(axis='y')

    # Manually adding the Method "Brackets" using text annotations
    methods = [r'Softmax', r'Balanced\ Softmax', r'CBS\ (Ours)']
    for i, method in enumerate(methods):
        ax.text(i * 3 + 1.5, -0.14, r'$\underbrace{\qquad \qquad \qquad}_{\mathrm{' + method + '}}$',
                ha='center', va='center', size=15,  # Fontsize 20 scales the brace itself.
                color='black', transform=ax.get_xaxis_transform(), clip_on=False)
        # Draw a line/bracket under the method name
        ax.plot([i * 3 + 3, i * 3 + 3], [1, -0.02], color='gray',
                lw=1, transform=ax.get_xaxis_transform(), clip_on=False)

    # plt.title('Global Gradient Balance Overview: Relative Difference (%)', pad=40, fontsize=15)
    plt.tight_layout()
    save_fig("grad_balance_overview.pdf")
    plt.show()


def visualise_preference_issue():
    # 1. Setup the Data
    data = {
        'Dataset': ['C10-LT10', 'C10-LT50', 'C10-LT100', 'C100-LT10',
                    'C100-LT50', 'C100-LT100', 'ImageNet-LT', 'Places-LT', 'iNaturalist', 'LVIS'],
        'Softmax': [10.04, 29.40, 41.41, 31.95, 59.90, 65.25, 103.27, 77.82, 16.07, 125.84],
        'Balanced Softmax (state-of-the-art)': [9.25, 14.74, 20.96, 13.94, 25.89, 32.08, 43.63, 24.13, 1.80, 97.72],
        # 'Balanced Softmax (state-of-the-art)': [8.30, 15.76, 22.28, 15.92, 28.61, 31.35, 42.58, 25.98, 1.95, 102.82],
        'CBS (ours)': [8.38, 12.24, 12.62, 12.36, 21.58, 31.13, 35.73, 9.67, 0.68, 86.79]
        # 'CBS (ours)': [7.92, 12.19, 9.90, 14.26, 19.40, 30.67, 34.98, 10.50, 0.75, 71.03]
    }

    df = pd.DataFrame(data)
    df_melted = df.melt(id_vars='Dataset', var_name='Method', value_name='Metric Value')

    plt.figure()
    sns.set_style("whitegrid")

    # 3. Create Bar Chart
    # Using a "Paired" or "Muted" palette for a professional look
    ax = sns.barplot(data=df_melted, x='Dataset', y='Metric Value', hue='Method',
                     palette=['#4C72B0', '#DD8452', '#55A868'])

    # 4. Customise Axis and Labels
    ax.set_ylabel('Model Imbalance Level (lower is better)')
    # Adjusting tick label sizes
    plt.xticks(rotation=15)
    plt.xlabel("")

    # Customise Legend
    plt.legend(loc='upper left')
    # 5. Final Polish
    sns.despine()  # Removes top and right borders
    plt.tight_layout()
    save_fig('model_imbalance_level.pdf')
    plt.show()


def cbs_bs_box_plot(data="int_lvis"):
    # Global styling
    sns.set_theme(style="ticks", font="Arial")  # 'ticks' is cleaner for journals
    plt.rcParams['pdf.fonttype'] = 42  # Ensures text is editable in PDF
    import matplotlib.ticker as ticker
    # Bar plot colours
    journal_palette = {"Balanced Softmax": "#dd8452", "CBS": "#55a868"}
    grid_color = "#e0e0e0"  # Standard light grey grid colour
    line_width = 0.8

    def style_boxplot(data_list1, data_list2, title, ax, y_lim):
        t_stat, p_val = stats.ttest_ind(data_list2, data_list1,
                                        equal_var=False,
                                        alternative='greater')
        p_val = round(p_val, 4)
        p_val = rf"$p = {p_val}$"
        data_list = data_list1 + data_list2
        df = pd.DataFrame({'Method': ['Balanced Softmax'] * 5 + ['CBS'] * 5, 'Recall': data_list})

        # Plot boxes: 'edgecolor' makes the border slightly darker than the fill for clarity
        sns.boxplot(x='Method', y='Recall', hue='Method', data=df, ax=ax,
                    palette=journal_palette, width=0.4,
                    linewidth=1.2,  # Keeps the box borders defined
                    showcaps=True, legend=False)

        # 1. Set the Spine (Axis) width
        for spine in ['left', 'bottom']:
            ax.spines[spine].set_linewidth(line_width)
            ax.spines[spine].set_color(grid_color)

        # 2. Set the Tick width to match
        # 'direction="inout"' or 'direction="out"' ensures they don't look bulky inside the plot
        ax.tick_params(axis='both',
                       which='both',
                       width=line_width,  # <--- This must match line_width
                       color=grid_color,  # <--- This must match grid_color
                       length=4)  # Adjust length if they look too long
        ax.yaxis.set_major_locator(ticker.MultipleLocator(1.0))
        # 3. Set the Grid width to match
        ax.yaxis.grid(True,
                      linestyle='-',
                      linewidth=line_width,  # <--- This must match line_width
                      color=grid_color,
                      alpha=1.0)

        # # 3. Y-Axis Grid
        # ax.yaxis.grid(True, linestyle='-', which='major', color=grid_color, alpha=1.0)
        ax.set_axisbelow(True)

        # 4. Refined labels and title
        ax.set_title(title, color='#333333')
        ax.set_ylabel('Testing recall', color='#333333')
        ax.set_ylim(y_lim)
        ax.set_xlabel('')

        # 5. Annotation
        ax.text(0.5, y_lim[1] - 0.4, p_val, ha='center', color='#555555')

        sns.despine(ax=ax)  # This removes top and right spines

        # Optional: If you want the bottom axis to disappear completely like the grid:
        ax.spines['bottom'].set_visible(True)
        ax.xaxis.grid(False)

    if data == "imagenet_place":
        imagenet_bs = [50.99, 52.32, 51.00, 49.47, 50.18]
        imagenet_cbs = [52.00, 52.52, 52.62, 51.80, 51.35]
        place_bs = [28.96, 29.41, 29.19, 29.85, 28.77]
        place_cbs = [30.65, 30.73, 30.17, 30.76, 30.04]

        fig, (ax1, ax2) = plt.subplots(1, 2)
        # Execute styling for both
        style_boxplot(imagenet_bs, imagenet_cbs, 'ImageNet-LT',
                      ax1, (49, 53))
        style_boxplot(place_bs, place_cbs, 'Place-LT',
                      ax2, (28, 32))
    elif data == "int_lvis":
        int_bs = [69.61, 69.72, 69.13, 69.45, 69.30]
        int_cbs = [69.67, 69.75, 70.89, 70.78, 69.91]
        lvis_bs = [24.22, 24.51, 24.27, 24.19, 24.35]
        lvis_cbs = [25.03, 25.08, 25.23, 24.92, 24.52]

        fig, (ax1, ax2) = plt.subplots(1, 2)
        style_boxplot(int_bs, int_cbs, 'iNaturalist2018', ax1, (67, 72))
        style_boxplot(lvis_bs, lvis_cbs, 'LVIS', ax2, (24, 26))

    else:
        assert data == "cifar"
        fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2)
        c10_0_1_bs = [89.25, 88.50, 88.58, 89.07, 88.96]
        c10_0_02_bs = [81.93, 83.23, 82.44, 82.65, 82.55]
        c10_0_01_bs = [78.32, 78.53, 79.09, 79.65, 79.53]
        c100_0_1_bs = [60.16, 60.32, 59.96, 60.89, 60.35]
        c100_0_02_bs = [49.47, 50.97, 49.96, 50.09, 50.77]
        c100_0_01_bs = [45.83, 45.75, 45.77, 45.57, 44.99]

        c10_0_1_cbs = [89.24, 89.05, 89.56, 89.10, 89.43]
        c10_0_02_cbs = [83.97, 83.54, 83.86, 84.04, 84.28]
        c10_0_01_cbs = [81.58, 81.13, 81.44, 81.26, 81.73]
        c100_0_1_cbs = [61.07, 59.63, 60.28, 59.97, 60.89]
        c100_0_02_cbs = [51.54, 50.65, 51.00, 49.20, 51.14]
        c100_0_01_cbs = [45.14, 46.45, 46.36, 46.03, 45.92]

        style_boxplot(c10_0_1_bs, c10_0_1_cbs, 'C10-LT10', ax1, (88, 90))
        style_boxplot(c10_0_02_bs, c10_0_02_cbs, 'C10-LT50', ax2, (81, 85))
        style_boxplot(c10_0_01_bs, c10_0_01_cbs, 'C10-LT100', ax3, (78, 82))
        style_boxplot(c100_0_1_bs, c100_0_1_cbs, 'C100-LT10', ax4, (59, 62))
        style_boxplot(c100_0_02_bs, c100_0_02_cbs, 'C100-LT50', ax5, (49, 52))
        style_boxplot(c100_0_01_bs, c100_0_01_cbs, 'C100-LT100', ax6, (44, 47))

    plt.tight_layout()
    save_fig(f'bs_cbs_{data}_statistical.pdf')
    plt.show()
