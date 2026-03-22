import numpy as np
import torch
import torchvision.datasets
from PIL import Image
from torchvision import transforms
from utils.data_generation import modify_dataset, make_binary_dataset
from utils.utils import unpickle, get_num_data_per_class, change_to_coarse_labels, change_to_new_fine_labels

class_prob_paths = {
    '10': {
        '1.0': "saved/results_for_report/linear_layer/softmax/res_bias/2023-07-19_13-39-24_res32_cifar10_imbalance_exp_0_gpus1/class_prob.pk",
        '0.1': "saved/results_for_report/linear_layer/softmax/res_bias/2023-07-19_13-39-24_res32_cifar10_imbalance_exp_0-1_gpus1/class_prob.pk",
        '0.02': "saved/results_for_report/linear_layer/softmax/res_bias/2023-07-19_13-39-24_res32_cifar10_imbalance_exp_0-02_gpus1/class_prob.pk",
        '0.01': "saved/results_for_report/linear_layer/softmax/res_bias/2023-07-19_13-39-24_res32_cifar10_imbalance_exp_0-01_gpus1/class_prob.pk"
    },
    '100': {
        '1.0': "saved/results_for_report/linear_layer/softmax/res_bias/2023-07-19_13-39-24_res32_cifar100_imbalance_exp_0_gpus1/class_prob.pk",
        '0.1': "saved/results_for_report/linear_layer/softmax/res_bias/2023-07-19_13-39-24_res32_cifar100_imbalance_exp_0-1_gpus1/class_prob.pk",
        '0.02': "saved/results_for_report/linear_layer/softmax/res_bias/2023-07-19_13-39-24_res32_cifar100_imbalance_exp_0-02_gpus1/class_prob.pk",
        '0.01': "saved/results_for_report/linear_layer/softmax/res_bias/2023-07-19_13-39-24_res32_cifar100_imbalance_exp_0-01_gpus1/class_prob.pk"
    }
}


# image size 32*32
class CIFAR(object):

    def __init__(self, logger, dataset_function, data_loader_func, data_path='./saved/data/cifar/',
                 num_classes=None,
                 imbalance_factor=1.0, major=None, minor=None,
                 head_class_idx=None, med_class_idx=None, tail_class_idx=None,
                 one_vs_all=None, test_types=None,
                 get_test_dataset=True,
                 batch_size=None, num_workers=None,
                 img_max=None, imbalance_type=None, gpu_rank=None,
                 distributed_sampler=None,
                 coarse_labels=None, coarse_label_id=None,
                 hpc=None,
                 aug_type=None,
                 ):
        self.key = f"cifar{num_classes}-LT{round(1 / imbalance_factor)}"
        # if major is not None:
        #     if len(major) > 1:
        #         major = [*range(major[0], major[1] + 1)]
        #     if len(minor) > 1:
        #         minor = [*range(minor[0], minor[1] + 1)]
        if test_types is None:
            test_types = ['balanced']
        if num_classes == 2:
            head_class_idx = [0]
            tail_class_idx = [1]
        self.image_size = 32
        self.head_class_idx = head_class_idx
        self.med_class_idx = med_class_idx
        self.tail_class_idx = tail_class_idx
        self.num_classes = num_classes
        # if config.dataset == 'cifar10':
        #     dataset_function = torchvision.datasets.CIFAR10
        # elif config.dataset == 'cifar100':
        #     dataset_function = torchvision.datasets.CIFAR100
        # else:
        #     raise Exception('Dataset name is not supported.')
        mean_cifar10 = [0.4914, 0.4822, 0.4465]
        std_cifar10 = [0.2023, 0.1994, 0.2010]

        mean_cifar100 = [0.5071, 0.4867, 0.4408]
        std_cifar100 = [0.2673, 0.2564, 0.2762]

        train_mean = eval_mean = mean_cifar10 if num_classes == 10 else mean_cifar100
        train_std = eval_std = std_cifar10 if num_classes == 10 else std_cifar100
        # train_set = dataset_function(data_path, train=True, download=True,
        #                              transform=transforms.Compose([transforms.ToTensor()]))
        #
        # # computer the mean and std for normalization
        # train_mean = train_set.data.mean(axis=(0, 1, 2)) / 255
        # train_std = train_set.data.std(axis=(0, 1, 2)) / 255
        # # print(train_mean, train_std)
        #
        # eval_set = dataset_function(data_path, train=False, download=False,
        #                             transform=transforms.Compose([transforms.ToTensor()]))
        #
        # eval_mean = eval_set.data.mean(axis=(0, 1, 2)) / 255
        # eval_std = eval_set.data.std(axis=(0, 1, 2)) / 255
        # print(eval_mean, eval_std)

        # **************** processing the data
        # normalize data manually:
        # https://inside-machinelearning.com/en/why-and-how-to-normalize-data-object-detection-on-image-in-pytorch-part-1/#Normalize_Data_Manually

        # CIFAR-100 specific means
        train_transform = transforms.Compose([
            transforms.RandomCrop(self.image_size, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(train_mean, train_std),
        ])

        eval_transform = transforms.Compose([
            transforms.Compose([transforms.ToTensor()]),
            transforms.Compose([transforms.Normalize(eval_mean, eval_std)])
        ])

        # download the dataset
        # transform operations is not called here, it will be called when reading the image data from the dataset.
        self.train_dataset = dataset_function(root=data_path, train=True, download=True,
                                              transform=train_transform)

        self.test_dataset = dataset_function(root=data_path, train=False, download=False,
                                             transform=eval_transform) if get_test_dataset else None

        if num_classes == 2:
            make_binary_dataset(major=major, minor=minor, train_dataset=self.train_dataset,
                                val_dataset=self.test_dataset)
        else:
            original_img_max = len(self.train_dataset.data) / len(self.train_dataset.classes)

            if imbalance_factor < 1 or (img_max is not None and img_max < original_img_max):
                modify_dataset(self.train_dataset, imbalance_factor=imbalance_factor, imbalance_type=imbalance_type,
                               head_class_idx=head_class_idx, med_class_idx=med_class_idx,
                               tail_class_idx=tail_class_idx,
                               img_max=img_max)

            if one_vs_all is not None:
                all_classes = [*range(self.num_classes)]
                all_classes.remove(one_vs_all)
                one_class = [one_vs_all]
                make_binary_dataset(all_classes, one_class,
                                    balance_test='balanced' in test_types, train_dataset=self.train_dataset,
                                    val_dataset=self.test_dataset)
                self.head_class_idx = [0]
                self.tail_class_idx = [1]
                self.med_class_idx = None
                self.num_classes = 2

            elif coarse_labels or coarse_label_id is not None:
                message = f"the total number of class should be 100 instead of {self.num_classes}"
                assert num_classes == 100, message
                hierarchical_data = unpickle("saved/data/cifar/cifar-100-python/cifar100_hierarchical.pkl")
                self.head_class_idx = None
                self.med_class_idx = None
                self.tail_class_idx = None

                self.train_dataset.num_per_cls_dict = None
                self.test_dataset.num_per_cls_dict = None

                self.train_dataset.num_classes = None
                self.test_dataset.num_classes = None

                self.train_dataset.classes = None
                self.test_dataset.classes = None

                if coarse_labels:
                    self.num_classes = len(hierarchical_data['coarse_labels'])
                    self.class_to_idx = {_i: label for _i, label in
                                         enumerate(hierarchical_data['coarse_labels'])}
                    self.test_dataset.class_to_idx = self.train_dataset.class_to_idx = self.class_to_idx

                    self.train_dataset.targets, self.test_dataset.targets = change_to_coarse_labels(
                        hierarchical_data['fine2coarse'],
                        self.train_dataset.targets,
                        self.test_dataset.targets
                    )

                elif coarse_label_id is not None:
                    fine_labels_subset = hierarchical_data['coarse2fine'][coarse_label_id]
                    self.num_classes = len(fine_labels_subset)
                    self.class_to_idx = {_i: [label, hierarchical_data['fine_labels'][label]] for _i, label
                                         in enumerate(fine_labels_subset)}
                    self.test_dataset.class_to_idx = self.train_dataset.class_to_idx = self.class_to_idx

                    self.original_class_to_new_class_idx = {}
                    for idx, label in enumerate(fine_labels_subset):
                        self.original_class_to_new_class_idx[label] = idx

                    self.train_dataset.targets, self.train_dataset.data = change_to_new_fine_labels(
                        self.train_dataset.targets, self.train_dataset.data, coarse_label_id,
                        hierarchical_data['fine2coarse'],
                        self.original_class_to_new_class_idx)
                    self.test_dataset.targets, self.test_dataset.data = change_to_new_fine_labels(
                        self.test_dataset.targets, self.test_dataset.data, coarse_label_id,
                        hierarchical_data['fine2coarse'],
                        self.original_class_to_new_class_idx)

        if distributed_sampler is not None:
            self.distributed_sampler = distributed_sampler['func'](dataset=self.train_dataset,
                                                                   **distributed_sampler['args'])
        else:
            self.distributed_sampler = None

        self.train_dataloader = data_loader_func(
            self.train_dataset,
            batch_size=batch_size, shuffle=self.distributed_sampler is None,
            num_workers=num_workers, pin_memory=torch.cuda.is_available(), sampler=self.distributed_sampler
        )
        # is it the explanation why we need to use pin memory:
        # https://discuss.pytorch.org/t/when-to-set-pin-memory-to-true/19723

        # balance_sampler = ClassAwareSampler(train_dataset)
        # self.train_balance = torch.utils.data.DataLoader(
        #     train_dataset,
        #     batch_size=batch_size, shuffle=False,
        #     num_workers=num_works, pin_memory=True, sampler=balance_sampler)
        self.train_num_per_cls_dict, _, _ = get_num_data_per_class(self.train_dataset.targets, self.num_classes)
        self.sorted_id = np.array(range(self.num_classes))
        # head_max = max(self.train_num_per_cls_dict[self.head_class_idx])
        # head_min = min(self.train_num_per_cls_dict[self.head_class_idx])
        #
        # med_max = max(self.train_num_per_cls_dict[self.med_class_idx])
        # med_min = min(self.train_num_per_cls_dict[self.med_class_idx])
        #
        # tail_max = max(self.train_num_per_cls_dict[self.tail_class_idx])
        # tail_min = min(self.train_num_per_cls_dict[self.tail_class_idx])

        if logger is not None:
            logger.info(f'==============Train data==============', gpu_rank=gpu_rank)
            logger.info(f"Total classes: {self.num_classes}", gpu_rank=gpu_rank)
            logger.info(self.train_dataset.class_to_idx, gpu_rank=gpu_rank)
            logger.info(self.train_dataset.data.shape, gpu_rank=gpu_rank)
            logger.info(self.train_num_per_cls_dict, gpu_rank=gpu_rank)
            logger.info(f'Train batch num: {len(self.train_dataloader)}')

        if get_test_dataset:
            self.test_dataloader = data_loader_func(
                self.test_dataset,
                batch_size=batch_size, shuffle=False,
                num_workers=num_workers, pin_memory=True)
            self.test_num_per_cls_dict, _, _ = get_num_data_per_class(self.test_dataset.targets, self.num_classes)
            if logger is not None:
                logger.info(f'==============Test data==============', gpu_rank=gpu_rank)
                logger.info(self.test_dataset.class_to_idx, gpu_rank=gpu_rank)
                logger.info(self.test_dataset.data.shape, gpu_rank=gpu_rank)
                logger.info(self.test_num_per_cls_dict, gpu_rank=gpu_rank)
                logger.info(f'Test batch num: {len(self.test_dataloader)}')

            self.test_dataset = {test_types[0]: self.test_dataset}
            self.test_dataloader = {test_types[0]: self.test_dataloader}


def cifar10(logger, **kwargs):
    # {'airplane': 0, 'automobile': 1, 'bird': 2, 'cat': 3,
    #     # 'deer': 4, 'dog': 5, 'frog': 6, 'horse': 7, 'ship': 8, 'truck': 9}
    if 'num_classes' not in kwargs:
        kwargs['num_classes'] = 10

    if kwargs['num_classes'] != 2:
        if 'imbalance_type' in kwargs and kwargs['imbalance_type'] == 'step':
            kwargs.update({
                'head_class_idx': [*range(0, 5)],
                'tail_class_idx': [*range(5, 10)]
            })
        elif 'head_class_idx' not in kwargs:
            kwargs.update({
                'head_class_idx': [*range(0, 3)],
                'med_class_idx': [*range(3, 6)],
                'tail_class_idx': [*range(6, 10)],
            })

    return CIFAR(logger=logger, dataset_function=ModifiedCIFAR10, **kwargs)


def cifar100(logger, **kwargs):
    if 'num_classes' not in kwargs:
        kwargs['num_classes'] = 100

    if kwargs['num_classes'] != 2:
        if 'imbalance_type' in kwargs and kwargs['imbalance_type'] == 'step':
            kwargs.update({
                'head_class_idx': [*range(0, 50)],
                'tail_class_idx': [*range(50, 100)]
            })
        elif 'head_class_idx' not in kwargs:
            kwargs.update({
                'head_class_idx': [*range(0, 35)],
                'med_class_idx': [*range(35, 70)],
                'tail_class_idx': [*range(70, 100)],
            })

    return CIFAR(logger=logger, dataset_function=ModifiedCIFAR100, **kwargs)


def get_item(use_transform, transform, data, targets, index):
    img, target = data[index], targets[index]
    # doing this so that it is consistent with all other datasets
    # to return a PIL Image
    img = Image.fromarray(img)
    if use_transform:
        if isinstance(transform, list):
            img = [t(img) for t in transform]
            target = [target] * len(img)
        else:
            img = transform(img)
    return img, target


class ModifiedCIFAR10(torchvision.datasets.CIFAR10):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.use_transform = self.transform is not None

    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        return get_item(self.use_transform, self.transform, self.data, self.targets, index)


class ModifiedCIFAR100(torchvision.datasets.CIFAR100):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.use_transform = self.transform is not None

    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        return get_item(self.use_transform, self.transform, self.data, self.targets, index)
