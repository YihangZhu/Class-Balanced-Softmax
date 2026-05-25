from collections import defaultdict
from copy import deepcopy

import numpy as np
import torch
from PIL import Image
from lvis import LVISEval, LVIS, LVISResults
from torchvision import transforms

from datasets.class_balanced_sampler import ClassBalancedSampler
from datasets.collate_fns import collate_fns
from utils.transforms import (ResizeLongestSide, pad_image2square, mask_img_func, crop_img_bbox, masks_sample_points,
                              crop_img_bbox_lvis)


class Modified_LVISEval(LVISEval):
    def __init__(self, **kwargs):
        super(Modified_LVISEval, self).__init__(**kwargs)

    def print_results(self, logger=None, gpu_rank=None):
        template = " {:<18} {} @[ IoU={:<9} | area={:>6s} | maxDets={:>3d} catIds={:>3s}] = {:0.3f}"

        for key, value in self.results.items():
            max_dets = self.params.max_dets
            if "AP" in key:
                title = "Average Precision"
                _type = "(AP)"
            else:
                title = "Average Recall"
                _type = "(AR)"

            if len(key) > 2 and key[2].isdigit():
                iou_thr = (float(key[2:]) / 100)
                iou = "{:0.2f}".format(iou_thr)
            else:
                iou = "{:0.2f}:{:0.2f}".format(
                    self.params.iou_thrs[0], self.params.iou_thrs[-1]
                )

            if len(key) > 2 and key[2] in ["r", "c", "f"]:
                cat_group_name = key[2]
            else:
                cat_group_name = "all"

            if len(key) > 2 and key[2] in ["s", "m", "l"]:
                area_rng = key[2]
            else:
                area_rng = "all"

            result = template.format(title, _type, iou, area_rng, max_dets, cat_group_name, value)
            if logger is None:
                print(result)
            else:
                logger.info(result, gpu_rank=gpu_rank)


class COCODataset(LVIS):
    def __init__(self, ann_file_path, img_path, target_size=None,
                 ignored_anns=None,
                 less_background=None,
                 mask_path=None,
                 mask_img=None,
                 masked_img_path=None,
                 mask_bbox_path=None,
                 object_detection=False,
                 crop_square=False,
                 num_prompts=None
                 ):
        self.object_detection = object_detection
        self.num_prompts = num_prompts
        self.crop_square = crop_square
        # self.prepare_modified_bbox = False
        self.mask_bbox_path = mask_bbox_path
        if ignored_anns:
            ignored_anns = torch.load(ignored_anns)
            print("ignored_anns loaded")
        self.ignored_anns = [ann['id'] for ann in ignored_anns] if ignored_anns else []
        self.mask_img = mask_img
        self.mask_path = mask_path
        super().__init__(ann_file_path)
        self.ann_ids = list(self.anns.keys())
        self.less_background = less_background
        self.img_path = img_path
        if target_size:
            self.resize_transform = ResizeLongestSide(target_size)
        self.num_per_cls_dict = None
        self.get_num_per_cls_dict()

        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        # self.random_flip_transform = transforms.RandomHorizontalFlip()
        self.to_tensor = transforms.ToTensor()
        self.normalise = transforms.Normalize(mean, std)
        self.masked_img_path = masked_img_path if mask_img else None

    def get_num_per_cls_dict(self):
        self.num_per_cls_dict = {}
        for key, value in self.cat_ann_map.items():
            self.num_per_cls_dict[key] = len(self.cat_ann_map[key])

    def get_head_med_tail_classes(self):
        head, med, tail = [], [], []
        for cat_id, cat in self.cats.items():
            if cat['frequency'] == 'f':
                head.append(cat_id - 1)
            elif cat['frequency'] == 'c':
                med.append(cat_id - 1)
            elif cat['frequency'] == 'r':
                tail.append(cat_id - 1)
            else:
                raise ValueError('The frequency type is invalid.')

        return head, med, tail

    def _create_index(self):
        self.logger.setLevel(20)
        self.logger.info("Creating index.....")
        self.img_ann_map = defaultdict(list)
        self.cat_img_map = defaultdict(list)
        self.cat_ann_map = defaultdict(list)
        # self.img_cat_overlap = defaultdict(set)

        self.anns = {}
        self.cats = {}
        self.imgs = {}

        # ignored_anns = []
        # for img in self.dataset["images"]:
        #     self.imgs[img["id"]] = img
        # for ann in self.dataset["annotations"]:
        #     mask = self.ann_to_mask(ann)
        #     if np.sum(mask) == 0:
        #         ignored_anns.append(ann)
        # torch.save(ignored_anns, "../saved/data/lvis/ignored_anns.pth")
        # exit()
        # if self.prepare_modified_bbox:
        #     self.modified_bbox = {}
        #     for img in self.dataset["images"]:
        #         self.imgs[img["id"]] = img
        #     from tqdm import tqdm
        #     for ann in tqdm(self.dataset["annotations"]):
        #         mask = self.ann_to_mask(ann)
        #         zero_min, zero_max, one_min, one_max = crop_black(mask)
        #         self.modified_bbox[ann['id']] = [one_min, zero_min, one_max - one_min, zero_max - zero_min]
        #     torch.save(self.modified_bbox, self.mask_bbox_path)
        #     self.logger.info('bboxes are prepared')
        #     exit()

        try:
            if self.mask_bbox_path is not None:
                modified_bbox = torch.load(self.mask_bbox_path, weights_only=True)
                for ann in self.dataset["annotations"]:
                    ann['bbox'] = modified_bbox[ann['id']]
                self.logger.info("Bboxes are updated.")
        except FileNotFoundError:
            self.logger.info(f"Mask bbox path not found: {self.mask_bbox_path}")

        for ann in self.dataset["annotations"]:
            if ann['id'] in self.ignored_anns:
                ann['ignore'] = 1
            else:
                self.img_ann_map[ann["image_id"]].append(ann)
                self.cat_ann_map[ann['category_id']].append(ann['id'])
                self.cat_img_map[ann["category_id"]].append(ann["image_id"])
                self.anns[ann["id"]] = ann
            if self.mask_path is not None:
                del ann['segmentation']
                del ann['area']
        for img in self.dataset["images"]:
            if len(self.img_ann_map[img['id']]) > 0:
                self.imgs[img["id"]] = img
                del img['flickr_url']

        for cat in self.dataset["categories"]:
            self.cats[cat["id"]] = cat

    def get_img_mask(self, idx, get_mask=True):
        ann_id = self.ann_ids[idx]
        ann = self.anns[ann_id]
        img = self.imgs[ann['image_id']]
        # cat_id = notation["category_id"]
        # img = coco.loadImgs(ids=[388464])
        # img = img[0]
        # load and display image
        # I = io.imread('%s/images/%s/%s'%(dataDir,dataType,img['file_name']))
        # use url to load image
        img_name = '/'.join(img['coco_url'].split('/')[-2:])
        img_path = f"{self.img_path}/{img_name}"
        image = Image.open(img_path).convert('RGB')
        if get_mask:
            if self.mask_path is not None:
                mask = np.array(Image.open(f"{self.mask_path}/{ann_id}.png"))
            else:
                mask = self.ann_to_mask(ann)
            if np.sum(mask) == 0:
                print("All values in the mask are 0")
            return image, mask, ann
        else:
            return image, None, ann

    def __getitem__(self, idx):
        if self.object_detection:
            img, mask, ann = self.get_img_mask(idx, get_mask=True)
            img = np.array(img)
            og_h, og_w, _ = img.shape
            img = self.resize_transform.apply_image(img, return_array=False)
            img = self.to_tensor(img)
            img = self.normalise(img)
            mask = torch.tensor(mask, dtype=torch.float)
            org_points = masks_sample_points(mask, k=self.num_prompts)
            points = self.resize_transform.apply_coords_torch(org_points, (og_h, og_w))
            points = points.unsqueeze(1)

            point_labels = torch.ones((points.shape[0], 1))
            target = ann['category_id'] - 1
            return ({'image': img,
                     'point_coords': points,
                     'org_points': org_points,
                     'point_labels': point_labels,
                     'original_size': (og_h, og_w)
                     },
                    {'mask': mask, "class": target},
                    idx)
        else:
            if self.masked_img_path is not None:
                ann = self.anns[self.ann_ids[idx]]
                img = np.array(Image.open(f"{self.masked_img_path}/{ann['id']}.png"))
                whole_img, ann = self.get_img_mask(idx, get_mask=False)
            else:
                whole_img, mask, ann = self.get_img_mask(idx, get_mask=self.mask_img)
                whole_img = np.array(whole_img)
                if self.mask_img:
                    img = mask_img_func(whole_img, mask)
                else:
                    img = whole_img
            if self.less_background:
                img = crop_img_bbox_lvis(img=img, xywh=ann['bbox'], square=self.crop_square)

            def pre_process(_img):
                _img = self.resize_transform.apply_image(_img, return_array=False)
                # _img = self.random_flip_transform(_img)
                _img = self.to_tensor(_img)
                _img = pad_image2square(_img)
                _img = self.normalise(_img)
                return _img

            img = pre_process(img)
            # whole_img = pre_process(whole_img)
            target = ann['category_id'] - 1  # in the model class index starts from zero instead of one
            return img, target, idx

    def __len__(self):
        return len(self.ann_ids)

    def info(self):
        return f'total_class_num: {len(self.cats)}, total_data: {len(self.anns)}, ' \
               f'sample_per_class: {min(self.num_per_cls_dict.values())}-{max(self.num_per_cls_dict.values())}'


notations_files = {
    'lvis_train': 'lvis_v1_train.json',
    'lvis_test': 'lvis_v1_val.json'
}

# mask_files = {
#     'lvis_train': '/scratch/groups/su004-neuralnet/datasets/lvis/masks/train'
#     if torch.cuda.is_available() else None,
#     'lvis_test': '/scratch/groups/su004-neuralnet/datasets/lvis/masks/test'
#     if torch.cuda.is_available() else None
# }

# masked_img_files = {
#     'lvis_train': '/scratch/groups/su004-neuralnet/datasets/lvis/masked_images/train'
#     if torch.cuda.is_available() else None,
#     'lvis_test': '/scratch/groups/su004-neuralnet/datasets/lvis/masked_images/test'
#     if torch.cuda.is_available() else None
# }

ignored_ann_file = {
    'lvis_train': 'saved/data/lvis/ignored_anns_train.pth',
    'lvis_test': 'saved/data/lvis/ignored_anns_val.pth'
}
# the modified bbox are generated from masks, more precise.
modified_bbox_file = {
    'lvis_train': 'saved/data/lvis/mask_bbox_train.pk',
    'lvis_test': 'saved/data/lvis/mask_bbox_test.pk'
}

mean_std = {
    'lvis': {'mean': [103.53, 116.28, 123.675], 'std': [1.0, 1.0, 1.0]},  # these values are from imagenet
}


class DataMaker:
    def __init__(self, logger=None, data_loader_func=None, distributed_sampler=None, key=None,
                 batch_size=None, num_workers=None, gpu_rank=None, train_sampler='', hpc=None, target_size=None,
                 collate_fn=None, **kwargs):
        test_type = 'imbalanced'
        if torch.cuda.is_available():
            self.root_path = "/scratch/groups/su004-neuralnet/datasets"
        else:
            self.root_path = "/Users/yihang/Datasets"
            batch_size = 2
        self.key = key
        self.image_size = target_size
        collate_function = collate_fns[collate_fn] if collate_fn is not None else None
        self.train_dataset = COCODataset(f"{self.root_path}/{key}/{notations_files[f'{key}_train']}",
                                         f"{self.root_path}/coco",
                                         ignored_anns=ignored_ann_file[f"{key}_train"],
                                         # mask_path=mask_files[f'{key}_train'],
                                         # masked_img_path=masked_img_files[f'{key}_train'],
                                         mask_bbox_path=modified_bbox_file[f'{key}_train'],
                                         target_size=target_size,
                                         **kwargs)
        self.head_class_idx, self.med_class_idx, self.tail_class_idx = self.train_dataset.get_head_med_tail_classes()

        self.test_dataset = COCODataset(f"{self.root_path}/{key}/{notations_files[f'{key}_test']}",
                                        f"{self.root_path}/coco",
                                        ignored_anns=ignored_ann_file[f"{key}_test"],
                                        # mask_path=mask_files[f'{key}_test'],
                                        # masked_img_path=masked_img_files[f'{key}_test'],
                                        mask_bbox_path=modified_bbox_file[f'{key}_test'],
                                        target_size=target_size,
                                        **kwargs)
        self.test_dataloader = {test_type: data_loader_func(self.test_dataset,
                                                            batch_size=batch_size, shuffle=False,
                                                            num_workers=num_workers, pin_memory=True,
                                                            collate_fn=collate_function)}

        if distributed_sampler is not None:
            self.distributed_sampler = distributed_sampler['func'](dataset=self.train_dataset,
                                                                   **distributed_sampler['args'])
        else:
            self.distributed_sampler = None

        self.train_dataloader = data_loader_func(
            self.train_dataset,
            batch_size=batch_size, shuffle=self.distributed_sampler is None,
            num_workers=num_workers, pin_memory=torch.cuda.is_available(), drop_last=True,
            sampler=self.distributed_sampler, collate_fn=collate_function)

        self.train_num_per_cls_dict = self.train_dataset.num_per_cls_dict
        self.train_num_per_cls_dict = np.zeros(len(self.train_dataset.num_per_cls_dict))
        # convert the class index starting from 1 to from zero.
        for key, value in self.train_dataset.num_per_cls_dict.items():
            self.train_num_per_cls_dict[key - 1] = value

        self.num_classes = len(self.train_num_per_cls_dict)

        if 'class_balanced-fix' in train_sampler:
            self.train_sampling = ClassBalancedSampler(len(self.train_dataset.anns), self.num_classes,
                                                       self.train_dataset.cat_ann_map, "fix" in train_sampler)

        if logger is not None:
            logger.info('Data prepared.', gpu_rank=gpu_rank)
            logger.info(f'==============Data details==============', gpu_rank=gpu_rank)
            logger.info(f"Total classes: {self.num_classes}", gpu_rank=gpu_rank)
            logger.info(f'Train batch num: {len(self.train_dataloader)}, '
                        f'test batch num: {len(self.test_dataloader[test_type])}')
            logger.info(f"train {self.train_dataset.info()}", gpu_rank=gpu_rank)
            logger.info(f"test {self.test_dataset.info()}", gpu_rank=gpu_rank)
        self.results = {}

    def update_train_dataset(self):
        if hasattr(self, 'train_sampler'):
            self.train_dataset.ann_ids = self.train_sampling.generate_sample_ids()

    def record_results(self, output, ann_ids):
        _, pred = output.max(1)
        for cls, ann_id in zip(pred, ann_ids):
            if isinstance(ann_id, torch.Tensor):
                ann_id = ann_id.item()
            ann_id = self.test_dataset.ann_ids[ann_id]
            self.results[ann_id] = cls.item() + 1  # ann_id: predicted_class_id

    def evaluate(self, debug=False, logger=None, gpu_rank=None):
        del self.train_dataset
        delete_anns = []
        self.test_dataset = COCODataset(
            f"{self.root_path}/{self.key}/{notations_files[f'{self.key}_test']}",
            f"{self.root_path}",
            mask_bbox_path=modified_bbox_file[f'{self.key}_test'],
            ignored_anns=ignored_ann_file[f"{self.key}_test"]
        )
        lvis_results = deepcopy(self.test_dataset.anns)
        if debug:
            for ann in lvis_results.values():
                ann['score'] = 1
        else:
            for ann_id, cls in self.results.items():
                lvis_results[ann_id]['category_id'] = cls
                lvis_results[ann_id]['score'] = 1

        self.results = lvis_results
        for ann_id, ann in self.results.items():
            if 'score' not in ann:
                delete_anns.append(ann_id)
        for ann_id in delete_anns:
            del self.results[ann_id]
        lvis_results = LVISResults(self.test_dataset, list(self.results.values()))
        lvis_eval = Modified_LVISEval(lvis_gt=self.test_dataset, lvis_dt=lvis_results)
        lvis_eval.logger.setLevel(20)
        lvis_eval.run()
        lvis_eval.print_results(logger=logger, gpu_rank=gpu_rank)


def lvis_v1(**kwargs):
    return DataMaker(key='lvis', **kwargs)


if __name__ == '__main__':
    img_dir = '/Users/yihang/Datasets/coco'
    annFile = '/Users/yihang/Datasets/lvis/lvis_v1_train.json'
    key = "lvis"
    type = "train"
    # initialize COCO api for instance annotations
    train_dataset = COCODataset(annFile,
                                img_dir,
                                # ignored_anns=ignored_ann_file[f"{key}_{type}"],
                                # mask_path=mask_files[f'{key}_{type}'],
                                # masked_img_path=masked_img_files[f'{key}_{type}'],
                                # mask_bbox_path=modified_bbox_file[f'{key}_{type}'],
                                target_size=224,
                                mask_img=True,
                                less_background=True,
                                crop_square=True)
    percentages = []

    # np.random.shuffle(train_dataset.ann_ids)
    # for i in tqdm(range(100000)):
    #     image, mask, ann = train_dataset.get_img_mask(i, get_mask=True)
    #     image = np.array(image)
    #     image = crop_img_bbox(image, ann['bbox'], False)
    #     mask = crop_img_bbox(mask, ann['bbox'], False)
    #     non_zeros = np.sum(mask)
    #     percentage = non_zeros / (np.max(mask.shape) ** 2)
    #     percentages.append(percentage)
    # mean_percentage = np.mean(percentages)
    # print(mean_percentage)

    # from pathlib import Path
    # path = '/scratch/groups/su004-neuralnet/datasets/lvis/masked_images/train'
    # Path(path).mkdir(exist_ok=True, parents=True)
    # for idx in range(len(coco.anns)):
    #     img, mask, ann = coco.get_img_mask(idx, get_mask=True)
    #     img = mask_img_func(img, mask)
    #     Image.fromarray(img).save(f"{path}/{ann['id']}.png")
    # import matplotlib.pyplot as plt

    #
    # # I, ann = coco.get_img_mask(0, get_mask=False)
    # # I = I.permute(1, 2, 0).numpy()
    #

    I, mask, ann = train_dataset.get_img_mask(19312, get_mask=True)
    import matplotlib.pyplot as plt

    # load and display instance annotations
    plt.imshow(I)
    plt.axis('off')
    plt.show()

    I = crop_img_bbox(img=I, xywh=ann['bbox'], square=True)

    # I = np.array(I)
    # mask = 1 - mask

    # I = mask_img_func(I, mask)
    I = Image.fromarray(I)

    import matplotlib.pyplot as plt

    # load and display instance annotations
    plt.imshow(I)
    plt.axis('off')
    plt.show()

    print("Finished.")
