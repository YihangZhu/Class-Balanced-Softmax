import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from datasets.class_balanced_sampler import ClassBalancedSampler
from datasets.collate_fns import collate_fns
from utils.compute import augmentation_randncls_func, augmentation_sim_func, augmentation_randnclsstack_func
from utils.data_generation import modify_dataset_iip, save_dataset
from utils.transforms import crop_img_bbox
from utils.transforms import pad_image2square
from utils.utils import unpickle, get_num_data_per_class

key_inaturalist2018 = 'inaturalist2018'
key_imagenet_lt = 'imagenet_lt'
key_place_lt = 'place_lt'
key_imagenet_1k = 'imagenet_1k'
if 'SCRATCHDIR' not in os.environ:
    os.environ['SCRATCHDIR'] = ""

paths = {
    key_inaturalist2018: {
        'data': os.environ['SCRATCHDIR'] + '/data/inaturalist2018/',
        'train_num_per_cls_dict': "saved/data/iNaturalist18/train_num_per_cls_dict.pk",
        'train_txt': "saved/data/iNaturalist18/iNaturalist18_train.txt",
        'test_txt': {
            'balanced': "saved/data/iNaturalist18/iNaturalist18_val_balanced.txt",
            'imbalance_head_to_head': "saved/data/iNaturalist18/iNaturalist18_val_imbalance_head_to_head.txt",
            'imbalance_head_to_tail': "saved/data/iNaturalist18/iNaturalist18_val_imbalance_head_to_tail.txt"
        },
        'class_sets': 'saved/data/iNaturalist18/head_med_tail.pkl',
        'hierarchical': 'saved/data/iNaturalist18/iNaturalist2018_hierarchical.pkl',
        'class_prob': "saved/results_for_report/linear_layer/softmax/res_bias/2023-07-19_13-54-49_res50_iNaturalist2018_0_gpus4/class_prob.pk",
        'sorted_class_id': "saved/data/iNaturalist18/sorted_class_id.pk",
        'wds_shards': {
            'train': os.environ['SCRATCHDIR'] + "/data/inaturalist18_shards/train/train-{000000..000437}.tar",
            'test': os.environ['SCRATCHDIR'] + "/data/inaturalist18_shards/test/test-{000000..000024}.tar"

        }
    },
    key_imagenet_lt: {
        'data': os.environ['SCRATCHDIR'] + '/data/imagenet/ILSVRC/Data/CLS-LOC/',
        'train_num_per_cls_dict': "saved/data/ImageNet_LT/train_num_per_cls_dict.pk",
        'train_txt': "saved/data/ImageNet_LT/ImageNet_LT_train.txt",
        'test_txt': {
            'balanced': "saved/data/ImageNet_LT/ImageNet_LT_test.txt"
        },
        'val_txt': "saved/data/ImageNet_LT/ImageNet_LT_val.txt",
        'class_sets': 'saved/data/ImageNet_LT/head_med_tail.pkl',
        'class_prob': 'saved/results_for_report/linear_layer/softmax/res_bias/2023-07-22_17-41-33_res50_imagenet_lt_0_gpus4/class_prob.pk',
        'sorted_class_id': "saved/data/ImageNet_LT/sorted_class_id.pk",
        'bboxes': "saved/results_for_report/imagenet_bbox/merged/bbox_path_0.txt",
        'wds_shards': {
            'train': os.environ['SCRATCHDIR'] + "/data/imagenet_shards/train/train-{000000..000231}.tar",
            'test': os.environ['SCRATCHDIR'] + "/data/imagenet_shards/test/test-{000000..000099}.tar"
        }
    },
    key_place_lt: {
        'data': os.environ['SCRATCHDIR'] + '/data/place365/',
        'train_num_per_cls_dict': "saved/data/Places_LT/train_num_per_cls_dict.pk",
        'train_txt': "saved/data/Places_LT/Places_LT_train.txt",
        'test_txt': {'balanced': "saved/data/Places_LT/Places_LT_test.txt"},
        'val_txt': "saved/data/Places_LT/Places_LT_val.txt",
        'class_sets': 'saved/data/Places_LT/head_med_tail.pkl',
        'class_prob': 'saved/results_for_report/linear_layer/softmax/res_bias/2024-08-02_21-03-10_res152_place_lt_0_gpus4/class_prob.pk',
        'sorted_class_id': "saved/data/Places_LT/sorted_class_id.pk"
    },
    key_imagenet_1k: {
        'data': os.environ['SCRATCHDIR'] + '/data/imagenet/ILSVRC/Data/CLS-LOC/',
        'train_num_per_cls_dict': "saved/data/ImageNet_1K/train_num_per_cls_dict.pk",
        'train_txt': "saved/data/ImageNet_1K/ImageNet_train.txt",
        'test_txt': {
            'balanced': "saved/data/ImageNet_1K/ImageNet_val.txt"
        },
        'class_sets': 'saved/data/ImageNet_LT/head_med_tail.pkl',
        'class_prob': 'saved/results_for_report/linear_layer/softmax/res_bias/2023-07-24_19-37-15_res50_imagenet_1k_0_gpus4/class_prob.pk',
        'sorted_class_id': "saved/data/ImageNet_LT/sorted_class_id.pk"
    }
}


# Image statistics
def get_rgb_mean_std(key):
    if key_inaturalist2018 in key:
        RGB_statistics = {
            'mean': [0.466, 0.471, 0.380],
            'std': [0.195, 0.194, 0.192]
        }
    else:
        RGB_statistics = {
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225]
        }
    return RGB_statistics['mean'], RGB_statistics['std']


# Data transformation with augmentation
def get_data_transform(mode, rgb_mean, rbg_std, key, aug_type="", views=None):
    normalize = transforms.Normalize(rgb_mean, rbg_std)
    img_size = 224
    rgb_mean = (0.485, 0.456, 0.406)
    ra_params = dict(translate_const=int(img_size * 0.45),
                     img_mean=tuple([min(255, round(255 * x)) for x in rgb_mean]))
    augmentation_randncls = augmentation_randncls_func(img_size, ra_params, normalize)
    augmentation_randnclsstack = augmentation_randnclsstack_func(img_size, ra_params, normalize)
    augmentation_sim = augmentation_sim_func(img_size, normalize)

    dc_transform = transforms.Compose([
        pad_image2square,
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize(rgb_mean, rbg_std)
    ])
    dc_test_transform = transforms.Compose([
        transforms.Compose([
            pad_image2square,
            transforms.Resize((img_size, img_size))]),
        transforms.Compose([transforms.ToTensor()]),
        transforms.Compose([transforms.Normalize(rgb_mean, rbg_std)])
    ])

    imgnet_train_transform = transforms.Compose([
        transforms.RandomResizedCrop(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0),
        transforms.ToTensor(),
        transforms.Normalize(rgb_mean, rbg_std)
    ])
    test_transform = transforms.Compose([
        transforms.Compose([transforms.Resize(256), transforms.CenterCrop(img_size)]),
        transforms.Compose([transforms.ToTensor()]),
        transforms.Compose([transforms.Normalize(rgb_mean, rbg_std)])
    ])

    if mode == 'train':
        if aug_type != "":
            if "scl" in aug_type:
                _, aug1, _, aug2, _ = aug_type.split('_')
                transform_list = []

                def get_t(_str, num, t_list):
                    if 'randcls' in _str:
                        t_list += [transforms.Compose(augmentation_randncls) for _ in range(num)]
                    elif 'sim' in _str:
                        t_list += [transforms.Compose(augmentation_sim) for _ in range(num)]
                    elif 'randclastack' in _str:
                        t_list += [transforms.Compose(augmentation_randnclsstack) for _ in range(num)]
                    return t_list

                transform_list = get_t(aug1, views[0], transform_list)
                transform_list = get_t(aug2, views[1], transform_list)
                return transform_list

            elif aug_type == "data_cleaning":
                return dc_transform
            elif aug_type == "rotation":
                return ["rotation", dc_transform]

        if key == key_inaturalist2018:
            return transforms.Compose([
                transforms.RandomResizedCrop(img_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(rgb_mean, rbg_std)
            ])
        else:
            return imgnet_train_transform

    else:  # test or val
        if aug_type == 'data_cleaning':
            return dc_test_transform
        elif aug_type == 'rotation':
            return ["rotation", dc_test_transform]
        else:
            return test_transform


# Dataset
class LT_Dataset(Dataset):
    def __init__(self, root, txt, num_classes=None, transform=None, less_background=None, bboxes=None):
        self.img_path = []
        self.targets = []
        self.transform = transform
        self.use_transform = self.transform is not None
        self.less_background = less_background
        self.bboxes = []
        with open(txt) as f:
            for line in f:
                image_path, target = line.split()
                if 'ILSVRC2012_val' in image_path:
                    image_path = image_path.split('/')
                    del image_path[1]
                    image_path = '/'.join(image_path)
                image_path = os.path.join(root, image_path)
                target = int(target)
                if bboxes is not None:
                    img_name = image_path.split('/')[-1]
                    if img_name in bboxes:
                        bbox = bboxes[img_name]
                    else:
                        bbox = None
                    self.bboxes.append(bbox)

                self.img_path.append(image_path)
                self.targets.append(target)
        self.num_per_cls_dict, self.sample_id_per_class, self.sample_ids = None, None, None
        self.update_info(num_classes)

    def update_info(self, num_classes):
        self.num_per_cls_dict, self.sample_id_per_class, self.sample_ids = get_num_data_per_class(
            self.targets, num_classes
        )

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, index):
        sample_id = self.sample_ids[index]
        path = self.img_path[sample_id]
        label = self.targets[sample_id]
        sample = Image.open(path).convert('RGB')
        if self.less_background is not None and self.bboxes[sample_id] is not None:
            # from utils.visualize import show_img
            # show_img(sample)
            sample = crop_img_bbox(img=sample, xyxy=self.bboxes[sample_id], square=False, pad=self.less_background,
                                   return_img=True, xyxy_inclusive=True)

            # show_img(sample)
        if self.use_transform:
            assert self.transform is not None, "Transform is None in the dataset."
            sample, label = transform_samples((sample, label), self.transform)
        return sample, label, sample_id

    def info(self, num_classes):
        return f'total_class_num: {num_classes}, total_data: {len(self.targets)}, ' \
               f'sample_per_class: {min(self.num_per_cls_dict)}-{max(self.num_per_cls_dict)}'


def get_shot_idx(num_sample_per_classes, many_shot_thr=100, low_shot_thr=20):
    many_shot = []
    median_shot = []
    low_shot = []
    for i, num in enumerate(num_sample_per_classes):
        if num > many_shot_thr:
            many_shot.append(i)
        elif num < low_shot_thr:
            low_shot.append(i)
        else:
            median_shot.append(i)
    return many_shot, median_shot, low_shot


def transform_samples(data, transform):
    sample, label = data
    if isinstance(transform, list):
        if transform[0] == "rotation":
            samples = [transform[1](sample.rotate(angle, expand=True)) for angle in
                       (0, 90, 180, 270)]
            sample = samples
            label = [0, 1, 2, 3]
        else:
            sample = [t(sample) for t in transform]
            label = [label] * len(sample)
    else:
        sample = transform(sample)

    return sample, label


class DataMaker:
    def __init__(self, logger=None, data_loader_func=torch.utils.data.DataLoader, distributed_sampler=None,
                 train_sampler="",
                 num_classes=None, key=None,
                 batch_size=None, num_workers=2, gpu_rank=None, test_types=None,
                 coarse_labels=None, coarse_label_id=None,
                 generate_testing_data=False, imbalance_factor=None, saved_file=None, aug_type="",
                 combine_train_test=None, minor=None, collate_fn=None, less_background=None,
                 use_wds=False):
        self.test_types = test_types
        if self.test_types is None:
            self.test_types = ['balanced']
        self.num_classes = num_classes
        unsupervised = False
        # if aug_type == "rotation":
        #     self.num_classes = 4
        #     unsupervised = True
        if less_background is not None:
            bboxes = {}
            with open(paths[key]['bboxes'], 'r') as f:
                for line in f:
                    elements = line.split('\t')
                    bboxes[elements[0]] = eval(elements[2])
        else:
            bboxes = None
        self.key = key
        self.image_size = 224
        if not torch.cuda.is_available():
            batch_size = 2
        data_path = paths[key]['data']

        self.train_num_per_cls_dict = torch.load(paths[key]['train_num_per_cls_dict'], weights_only=False)
        self.num_train_batch = sum(self.train_num_per_cls_dict) // batch_size
        if distributed_sampler is not None:
            self.num_train_batch /= distributed_sampler['args']['num_replicas']
        self.num_train_batch = int(self.num_train_batch)

        rgb_mean, rgb_std = get_rgb_mean_std(key)
        self.views = []
        if 'scl' in aug_type:
            _, _, num1, _, num2 = aug_type.split('_')
            self.views = [int(num1), int(num2)]

        self.transform_train = get_data_transform('train', rgb_mean, rgb_std, key, aug_type=aug_type,
                                                  views=self.views)
        self.transform_test = get_data_transform('test', rgb_mean, rgb_std, key, aug_type=aug_type)
        collate_function = collate_fns[collate_fn] if collate_fn is not None else None

        if coarse_labels or coarse_label_id is not None:
            hierarchical_data = unpickle(
                paths[key]['hierarchical']
            ) if (coarse_labels is not None or coarse_label_id is not None) else None

            self.original_class_to_new_class_idx = {label: idx for idx, label in
                                                    enumerate(hierarchical_data['coarse2fine'][coarse_label_id])
                                                    } if coarse_label_id is not None else None

            self.head_class_idx = None
            self.med_class_idx = None
            self.tail_class_idx = None

            if coarse_labels:
                self.num_classes = len(hierarchical_data['coarse_labels'])
            elif coarse_label_id is not None:
                self.num_classes = len(hierarchical_data['coarse2fine'][coarse_label_id])
                self.class_to_idx = {_i: [label] for _i, label
                                     in enumerate(hierarchical_data['coarse2fine'][coarse_label_id])}
        else:
            class_sets = torch.load(paths[key]['class_sets'], weights_only=True)
            self.head_class_idx = class_sets['head']
            self.med_class_idx = class_sets['medium']
            self.tail_class_idx = class_sets['tail']

        if use_wds:
            raise NotImplementedError
        else:
            self.train_dataset = LT_Dataset(data_path, paths[key]['train_txt'],
                                            num_classes=None if unsupervised else self.num_classes,
                                            transform=self.transform_train, less_background=less_background,
                                            bboxes=bboxes)
            if 'class_balanced' in train_sampler:
                self.train_sampler = ClassBalancedSampler(len(self.train_dataset.targets), self.num_classes,
                                                          self.train_dataset.sample_id_per_class,
                                                          fix_total_num="fix" in train_sampler)
                self.update_train_dataset()
            # self.head_class_idx, self.med_class_idx, self.tail_class_idx = get_shot_idx(self.train_dataset.num_per_cls_dict)
            # class_sets = {
            #     'head': self.head_class_idx,
            #     'medium': self.med_class_idx,
            #     'tail': self.tail_class_idx
            # }
            # torch.save(class_sets, paths[key]['class_sets'])
            # exit()

            if generate_testing_data:
                test_dataset = LT_Dataset('', paths[key]['test_txt'][self.test_types[0]],
                                          num_classes=None if unsupervised else self.num_classes,
                                          bboxes=bboxes)
                modify_dataset_iip(test_dataset, self.num_classes, imbalance_factor,
                                   img_max=test_dataset.num_per_cls_dict[0]
                                   # head_class_idx=self.tail_class_idx,
                                   # med_class_idx=self.med_class_idx,
                                   # tail_class_idx=self.head_class_idx
                                   )
                save_dataset(test_dataset, saved_file)
                return
            else:
                self.test_dataset = dict()
                self.test_dataloader = dict()
                for test_type in self.test_types:
                    self.test_dataset[test_type] = LT_Dataset(
                        data_path, paths[key]['test_txt'][test_type],
                        num_classes=None if unsupervised else self.num_classes,
                        transform=self.transform_test,
                        bboxes=bboxes
                    )
                    if combine_train_test:
                        self.train_dataset.img_path.extend(self.test_dataset[test_type].img_path)
                        self.train_dataset.targets.extend(self.test_dataset[test_type].targets)
                        self.train_dataset.update_info(num_classes)

                    self.test_dataloader[test_type] = data_loader_func(
                        self.test_dataset[test_type],
                        batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=torch.cuda.is_available(), collate_fn=collate_function)

                if minor is not None:
                    self.head_class_idx = [0]
                    self.med_class_idx = []
                    self.tail_class_idx = [1]
                    self.num_classes = 2

                    new_img_path = []
                    rest_ids = []
                    for i, t in enumerate(self.train_dataset.targets):
                        if t == minor:
                            new_img_path.append(self.train_dataset.img_path[i])
                        else:
                            rest_ids.append(i)

                    new_targets = [1] * len(new_img_path)

                    rest_sample_num = imbalance_factor * len(new_targets)

                    selected_ids = np.random.choice(rest_ids, rest_sample_num, replace=False)

                    rest_img_path = [self.train_dataset.img_path[t] for t in selected_ids]
                    rest_targets = [0] * len(selected_ids)

                    new_targets.extend(rest_targets)
                    new_img_path.extend(rest_img_path)

                    self.train_dataset.targets = new_targets
                    self.train_dataset.img_path = new_img_path
                    self.train_dataset.update_info(self.num_classes)
                    del self.test_dataset
                    del self.test_dataloader

            if distributed_sampler is not None:
                self.distributed_sampler = distributed_sampler['func'](dataset=self.train_dataset,
                                                                       **distributed_sampler['args'])
                logger.info('===> Using distributed sampler', gpu_rank=gpu_rank)
            else:
                self.distributed_sampler = None

            self.train_dataloader = data_loader_func(
                self.train_dataset,
                batch_size=batch_size, shuffle=self.distributed_sampler is None,
                num_workers=num_workers, pin_memory=torch.cuda.is_available(), drop_last=True,
                sampler=self.distributed_sampler, collate_fn=collate_function)

        # if not torch.cuda.is_available():
        if "sorted_class_id" in paths[key]:
            self.sorted_id = torch.load(paths[key]['sorted_class_id'], weights_only=False)
        # self.class_prob = torch.load(paths[key]['class_prob'],
        #                              map_location=torch.device('cpu') if not torch.cuda.is_available() else None,
        #                              weights_only=True)
        if logger is not None:
            logger.info('Data prepared.', gpu_rank=gpu_rank)
            logger.info(f'==============Data details==============', gpu_rank=gpu_rank)
            logger.info(f"Total classes: {self.num_classes}", gpu_rank=gpu_rank)

            logger.info(f'Train batch num: {self.num_train_batch}', gpu_rank=gpu_rank)
            if hasattr(self, 'test_dataloader') and len(self.test_types) > 0:
                logger.info(f'test batch num: {len(self.test_dataloader[self.test_types[0]])}', gpu_rank=gpu_rank)
            if not unsupervised:
                logger.info(f'Train: '
                            f'head class samples ({np.sum(self.train_num_per_cls_dict[self.head_class_idx])}, {np.sum(self.train_num_per_cls_dict[self.head_class_idx]) / np.sum(self.train_num_per_cls_dict)}) '
                            f'med class samples ({np.sum(self.train_num_per_cls_dict[self.med_class_idx])}, {np.sum(self.train_num_per_cls_dict[self.med_class_idx]) / np.sum(self.train_num_per_cls_dict)}) '
                            f'tail class samples ({np.sum(self.train_num_per_cls_dict[self.tail_class_idx])}, {np.sum(self.train_num_per_cls_dict[self.tail_class_idx]) / np.sum(self.train_num_per_cls_dict)})'
                            , gpu_rank=gpu_rank)
                if not use_wds:
                    logger.info(f"train {self.train_dataset.info(self.num_classes)}", gpu_rank=gpu_rank)
                if hasattr(self, 'test_dataloader'):
                    for test_type in self.test_types:
                        logger.info(f"test {test_type} {self.test_dataset[test_type].info(self.num_classes)}",
                                    gpu_rank=gpu_rank)
                if hasattr(self, 'val_dataset'):
                    logger.info(f"val {self.val_dataset.info(self.num_classes)}", gpu_rank=gpu_rank)

    def get_batch_num(self, prefix):
        if 'Train' in prefix:
            return self.num_train_batch
        else:
            assert 'Eval' in prefix
            return len(self.test_dataloader[self.test_types[0]])

    def update_train_dataset(self):
        if hasattr(self, 'train_sampler'):
            self.train_dataset.sample_ids = self.train_sampler.generate_sample_ids()


def inaturalist2018(**kwargs):
    return DataMaker(key=key_inaturalist2018, num_classes=8142, **kwargs)


def imagenet_lt(**kwargs):
    return DataMaker(key=key_imagenet_lt, num_classes=1000, **kwargs)


def place_lt(**kwargs):
    return DataMaker(key=key_place_lt, num_classes=365, **kwargs)


def imagenet_1k(**kwargs):
    return DataMaker(key=key_imagenet_1k, num_classes=1000, **kwargs)

# if __name__ == '__main__':
#     data_name = key_inaturalist2018
#     file_path = paths[data_name]['test_txt']['imbalance_random']
#     DataMaker(key=data_name, num_classes=8142, imbalanced_facter=1.0/3.0, generate_testing_data=True,
#               saved_file=file_path)
