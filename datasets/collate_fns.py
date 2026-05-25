import torch


def collate_fn(data):
    inputs = []
    targets = []
    idx = []
    for i in data:
        inputs.append(i[0])
        targets.append(i[1])
        idx.append(i[2])
    return inputs, targets, idx


def collate_fn_shard(data):
    samples, labels = data
    samples = [item for pair in zip(*samples) for item in pair]
    labels = [item for pair in zip(*labels) for item in pair]
    samples = torch.stack(samples, dim=0)
    labels = torch.tensor(labels)
    return samples, labels


def collate_tb_fn(data):
    inputs = []
    targets = []
    idx = []
    whole_imgs = []
    for i in data:
        inputs.append(i[0])
        targets.append(i[1])
        idx.append(i[2])
        whole_imgs.append(i[3])
    return {'objects': torch.stack(inputs, dim=0), 'whole_images': torch.stack(whole_imgs, dim=0)}, torch.tensor(
        targets), idx


def collate_bbox_fn(data):
    imgs = []
    point = []
    point_labels = []
    org_img_size = []
    masks = []
    boxes = []
    for k in data:
        i, j, _ = k
        imgs.append(i['image'])
        point.append(i['point'])
        point_labels.append(i['point_label'])
        org_img_size.append(i['org_img_size'])
        masks.append(j['mask'])
        boxes.append(j['bbox'])

    return ({'image': imgs,
             'point': point,
             'point_label': point_labels,
             'org_img_size': org_img_size},
            {'mask': masks, "bbox": boxes})


collate_fns = {
    'bbox': collate_bbox_fn,
    'object_detection': collate_fn,
    'tb_net': collate_tb_fn,
    'shard': collate_fn_shard
}
