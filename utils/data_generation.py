import numpy as np


def get_img_num_per_cls(cls_num, imb_factor, imb_type, img_max):
    img_num_per_cls = []
    if imb_type == 'exp':
        for cls_idx in range(cls_num):
            num = img_max * (imb_factor ** (cls_idx / (cls_num - 1.0)))
            img_num_per_cls.append(int(num))
            # this should be round(.), I keep it as int(.) to be consistent with the literature.
    else:
        assert isinstance(imb_type, list)
        assert imb_type[0] == 'step'

        head_class_num = int(imb_type[1])
        for cls_idx in range(head_class_num):
            img_num_per_cls.append(int(img_max))
        for cls_idx in range(cls_num - head_class_num):
            img_num_per_cls.append(int(img_max * imb_factor))
    return img_num_per_cls


def modify_dataset_iip(dataset, num_classes, imbalance_factor, img_max, head_class_idx=None, med_class_idx=None,
                       tail_class_idx=None):
    img_num_per_cls = np.zeros(num_classes, dtype=np.int64)
    for cls_idx in range(num_classes):
        num = round(img_max * (imbalance_factor ** (cls_idx / (num_classes - 1.0))))
        img_num_per_cls[cls_idx] = num
    # from collections import Counter
    # counted = Counter(img_num_per_cls)
    new_data = []
    new_targets = []
    if head_class_idx is not None:
        classes = head_class_idx.copy()
        classes += med_class_idx
        classes += tail_class_idx
    else:
        classes = [*range(num_classes)]
        np.random.shuffle(classes)
    targets = np.array(dataset.targets, dtype=np.int64)
    num_per_cls_dict = np.zeros(num_classes, dtype=np.int64)
    for the_class, the_img_num in zip(classes, img_num_per_cls):
        num_per_cls_dict[the_class] = the_img_num
        idx = np.where(targets == the_class)[0]
        np.random.shuffle(idx)
        select_idx = idx[:the_img_num]
        selected_data = [dataset.img_path[i] for i in select_idx]
        new_data.extend(selected_data)
        new_targets.extend([the_class, ] * the_img_num)

    dataset.img_path = new_data
    dataset.targets = new_targets
    dataset.num_per_cls_dict = num_per_cls_dict

    print(f'Dataset is generated: {dataset.info(num_classes)}')


def save_dataset(dataset, file):
    dataset_paths = []
    for path, target in zip(dataset.img_path, dataset.targets):
        dataset_path = ' '.join([path, str(target)])
        dataset_paths.append(dataset_path)
    with open(file, 'w') as f:
        f.write('\n'.join(dataset_paths))
    print(f'The dataset has been saved to {file}.')


def modify_dataset(dataset, imbalance_factor, imbalance_type,
                   head_class_idx, med_class_idx, tail_class_idx, img_max=None):
    if img_max is None:
        img_max = len(dataset.data) / len(dataset.classes)

    img_num_per_cls = get_img_num_per_cls(len(dataset.classes), imbalance_factor, imbalance_type, img_max)
    new_data = []
    new_targets = []
    targets_np = np.array(dataset.targets, dtype=np.int64)
    classes = head_class_idx.copy()
    if med_class_idx is not None:
        classes += med_class_idx
    classes += tail_class_idx
    assert len(classes) == len(dataset.classes)
    # np.random.shuffle(classes)
    dataset.num_per_cls_dict = np.zeros(len(dataset.classes), dtype=np.int64)
    for the_class, the_img_num in zip(classes, img_num_per_cls):
        idx = np.where(targets_np == the_class)[0]
        np.random.shuffle(idx)
        select_idx = idx[:the_img_num]
        new_data.append(dataset.data[select_idx, ...])
        new_targets.extend([the_class, ] * len(select_idx))
        dataset.num_per_cls_dict[the_class] = len(select_idx)
    new_data = np.vstack(new_data)
    dataset.data = new_data
    dataset.targets = new_targets
    assert len(dataset.data) == len(dataset.targets)


def make_binary_dataset(major, minor, balance_test=True, train_dataset=None, val_dataset=None):
    train_dataset.class_to_idx = {0: 'major', 1: 'minor'}
    train_dataset.classes = ['class_0', 'class_1']

    major_class_id = 0
    minor_class_id = 1

    # setup training data
    targets_np = np.array(train_dataset.targets, dtype=np.int64)
    new_train_data, new_train_targets = make_both_classes(
        targets_np=targets_np,
        minor=minor, major=major,
        minor_class_id=minor_class_id, major_class_id=major_class_id,
        old_dataset=train_dataset)
    train_dataset.data = new_train_data
    train_dataset.targets = new_train_targets

    if val_dataset is not None:
        # setup testing data
        val_dataset.class_to_idx = train_dataset.class_to_idx
        val_dataset.classes = train_dataset.classes
        targets_np = np.array(val_dataset.targets, dtype=np.int64)
        new_val_data, new_val_targets = make_both_classes(
            targets_np=targets_np,
            minor=minor, major=major,
            minor_class_id=minor_class_id, major_class_id=major_class_id,
            old_dataset=val_dataset,
            balance_test=balance_test)
        val_dataset.data = new_val_data
        val_dataset.targets = new_val_targets


def make_class(targets_np, class_list, class_id, old_dataset, new_data, new_targets, balanced=None, num_sample=None):
    data_temp = []
    for c in class_list:
        obj_ids = np.where(targets_np == c)[0]
        data_temp.extend(old_dataset.data[obj_ids, ...])
    if balanced:
        data_temp = data_temp[:num_sample]
    new_data.extend(data_temp)
    new_targets.extend([class_id] * len(data_temp))
    return len(data_temp)


def make_both_classes(targets_np, minor, major, minor_class_id, major_class_id, old_dataset, balance_test=None):
    new_data = []
    new_targets = []
    num_sample = make_class(targets_np, minor, minor_class_id, old_dataset, new_data, new_targets)
    make_class(targets_np, major, major_class_id, old_dataset, new_data, new_targets, balance_test, num_sample)
    new_data = np.stack(new_data, axis=0)
    return new_data, new_targets
