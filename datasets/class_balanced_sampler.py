"""Copyright (c) Facebook, Inc. and its affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.

Portions of the source code are from the OLTR project which
notice below and in LICENSE in the root directory of
this source tree.

Copyright (c) 2019, Zhongqi Miao
All rights reserved.
"""

import numpy as np


class ClassBalancedSampler:
    def __init__(self, num_samples, num_classes, sample_id_per_class, fix_total_num):
        # num_samples is the total number of samples in the dataset,
        # so that when using this balanced sampler the total number of samples remains the same.
        self.num_classes = num_classes
        self.sample_id_per_class = sample_id_per_class
        self.num_samples = num_samples
        if fix_total_num:
            self.num_sample_per_class = int(np.floor(num_samples / num_classes))
            diff = num_samples - self.num_sample_per_class * num_classes
            self.num_sample_per_class = np.ones(num_classes, dtype=int) * self.num_sample_per_class
            selected_classes = np.random.choice(range(num_classes), size=int(diff))
            for i in selected_classes:
                self.num_sample_per_class[i] += 1
            total_generated_samples = np.sum(self.num_sample_per_class)
            assert total_generated_samples == self.num_samples
        else:
            self.num_sample_per_class = np.ones(self.num_classes, dtype=int) * max(
                [len(i) for i in self.sample_id_per_class])
            # print(self.num_sample_per_class[0])

    def generate_sample_ids(self):
        new_sample_ids = []

        for class_id in range(self.num_classes):
            selected_anns = np.random.choice(self.sample_id_per_class[class_id],
                                             size=self.num_sample_per_class[class_id],
                                             replace=True)
            new_sample_ids.extend(selected_anns)

        return new_sample_ids

    # This is simply a random generator
# class RandomCycleIter:
#
#     def __init__(self, data, test_mode=False):
#         self.data_list = list(data)
#         self.length = len(self.data_list)
#         self.i = self.length - 1
#         self.test_mode = test_mode
#         self.indices = list(range(self.length))
#         self.random_generator = None
#
#     def __iter__(self):
#         return self
#
#     def update_rand_generator(self, g):
#         self.random_generator = g
#
#     def __next__(self):
#         self.i += 1
#         if self.i == self.length:
#             self.i = 0
#             if not self.test_mode:
#                 self.indices = torch.randperm(self.length, generator=self.random_generator).tolist()
#                 self.random_generator = None
#
#         return self.data_list[self.indices[self.i]]
#
#
# def class_aware_sample_generator(cls_iter, data_iter_list, num_samples, random_generator, num_replicas=1, rank=0):
#     i = 0
#     while i < num_samples:
#         cls_iter.update_rand_generator(random_generator)
#         class_id = next(cls_iter)
#         sample_generator = data_iter_list[class_id]
#         sample_generator.update_rand_generator(random_generator)
#         sample_generator = zip(*[sample_generator] * num_replicas)
#         temp_tuple = next(sample_generator)
#         yield temp_tuple[rank]
#         i += 1
#
#     assert i == num_samples


# # oversampling by duplicating samples in a class so that each class will generate the same total number of samples
# # comparing to the class that have the maximum number of samples.
# # this is controlled by self.num_samples
# class ClassBalancedSampler:
#     def __init__(self, dataset, seed=0):
#         self.seed = seed
#         self.g = torch.Generator()
#         self.g.manual_seed(self.seed)
#         num_classes = len(np.unique(dataset.targets))
#         self.class_iter = RandomCycleIter(range(num_classes))
#         cls_data_list = [list() for _ in range(num_classes)]
#         for i, label in enumerate(dataset.targets):
#             cls_data_list[label].append(i)
#         self.data_iter_list = [RandomCycleIter(x) for x in cls_data_list]
#         self.num_samples = max([len(x) for x in cls_data_list]) * len(cls_data_list)
#
#     def __iter__(self):
#         return class_aware_sample_generator(self.class_iter, self.data_iter_list,
#                                             self.num_samples, random_generator=self.g)
#
#     def __len__(self):
#         return self.num_samples


# class DistributedClassBalancedSampler(DistributedSampler):
#     def __init__(self, dataset, num_replicas=None, rank=None, seed=0):
#         super().__init__(dataset=dataset, num_replicas=num_replicas, rank=rank, shuffle=False, seed=seed,
#                          drop_last=False)
#         num_classes = len(np.unique(dataset.targets))
#         self.class_iter = RandomCycleIter(range(num_classes))
#         cls_data_list = [list() for _ in range(num_classes)]
#         for i, label in enumerate(dataset.targets):
#             cls_data_list[label].append(i)
#         self.data_iter_list = [RandomCycleIter(x) for x in cls_data_list]
#         self.total_size = max([len(x) for x in cls_data_list]) * len(cls_data_list)
#         assert self.total_size % self.num_replicas == 0
#         self.num_samples = math.ceil(self.total_size / self.num_replicas)
#
#     def __iter__(self):
#         g = torch.Generator()
#         g.manual_seed(self.seed + self.epoch)
#         return class_aware_sample_generator(self.class_iter, self.data_iter_list,
#                                             self.num_samples, random_generator=g,
#                                             num_replicas=self.num_replicas, rank=self.rank)

# if __name__ == '__main__':
#     from datasets.cifar import cifar10
#
#     train_data_sampler = {
#         'func': DistributedClassBalancedSampler,
#         'args':
#             {
#                 'num_replicas': 10,
#                 'rank': 8,
#                 'seed': 0
#             }
#     }
#     # dataset = inaturalist2018(logger=None, batch_size=256, num_workers=2, train_data_sampler=train_data_sampler,
#     #                           data_loader_func=torch.utils.data.DataLoader)
#     dataset = cifar10(logger=None, batch_size=128, num_workers=2, train_data_sampler=train_data_sampler,
#                       data_loader_func=torch.utils.data.DataLoader)
#
#     for i, (images, targets) in enumerate(dataset.train_dataloader):
#         print(i)
#     print(i)
#     print(len(images))
