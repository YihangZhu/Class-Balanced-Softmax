import io
import json
import os
import pickle
import random
from pathlib import Path

import PIL
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torchvision
from PIL import Image
from ruamel.yaml import YAML
from torchvision import transforms
from tqdm import tqdm


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def load_txt(path):
    content = []
    with open(path, "r") as f:
        for line in f.readlines():
            content.append(line.strip())
    return content


class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else:
            return super().find_class(module, name)


def unpickle(file, encoding="bytes"):
    with open(file, "rb") as fo:
        if not torch.cuda.is_available():
            try:
                file_picked = CPU_Unpickler(fo).load()
            except Exception as e:
                file_picked = torch.load(file, map_location=torch.device('cpu'), weights_only=True)
        else:
            file_picked = pickle.load(fo, encoding=encoding)
    return file_picked


def pickle_file(store_dir, file):
    with open(store_dir, 'wb') as fo:
        pickle.dump(file, fo)


def get_immediate_sub_folders(folder_dir):
    return next(os.walk(folder_dir))[1]


def get_immediate_files(folder_dir):
    return next(os.walk(folder_dir))[2]


def set_reproducibility(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def create_logger(file_dir, file_name):
    print('=> creating {}'.format(file_dir))
    file_dir.mkdir(parents=True, exist_ok=True)
    log_file = '{}.txt'.format(file_name)
    final_log_file = file_dir / log_file
    import logging
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(filename=str(final_log_file),
                        format=head)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    logging.getLogger('').addHandler(console)
    return logger


# logging in this way when using multiprocessing package.
class Logging:
    def __init__(self, file_dir, file_name):
        file_dir.mkdir(parents=True, exist_ok=True)
        log_file = '{}.txt'.format(file_name)
        self.log_file = file_dir / log_file

    def info(self, message, gpu_rank=0, console=True):
        # only log rank 0 GPU if running with multiple GPUs/multiple nodes.
        if gpu_rank == 0:
            if console:
                print(message)

        with open(f'{self.log_file}_{gpu_rank}.txt', 'a') as f:  # a for append to the end of the file.
            print(message, file=f)


class MultiEpochsDataLoader(torch.utils.data.DataLoader):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._DataLoader__initialized = False
        self.batch_sampler = _RepeatSampler(self.batch_sampler)
        self._DataLoader__initialized = True
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)


class _RepeatSampler(object):
    """ Sampler that repeats forever.
    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)


def count_parameters(model, only_trainable=False):
    if only_trainable:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())


def get_free_port():
    import socket
    sock = socket.socket()
    sock.bind(('', 0))
    free_port = sock.getsockname()[1]
    return free_port


def save_yaml(file, path):
    with open(path, 'wb') as f:
        yaml = YAML()
        yaml.default_flow_style = False
        yaml.dump(file, f)


def load_yaml(path):
    with open(path, 'rb') as f:
        yaml = YAML()
        dt = yaml.load(f)
        return dt


def check_file(file_path):
    path = Path(file_path)
    return path.is_file()


def check_dir(dir_path):
    path = Path(dir_path)
    return path.is_dir()


def print_current_dir():
    print(os.getcwd())


def retrieve_fc(self, classifier):
    """this function is only used to put fc layer back to the main model frame for loading the checkpoints from the
    previous version model"""
    if hasattr(classifier, 'fc'):
        self.fc = classifier.fc


class IsNewTrainEpoch(nn.Module):
    def __init__(self, epoch):
        super().__init__()
        self.running_val_set = False
        self.epoch = epoch

    def set_epoch(self, epoch):
        self.epoch = epoch

    def forward(self):
        if self.training:
            if self.running_val_set:
                self.running_val_set = False
                self.epoch += 1
                return True
        else:
            self.running_val_set = True
        return False


def get_attribute(obj, attr_name, ddp_on=False):
    if ddp_on:
        if hasattr(obj.module, f'{attr_name}'):
            return getattr(obj.module, f'{attr_name}')
    else:
        if hasattr(obj, f'{attr_name}'):
            return getattr(obj, f'{attr_name}')
    return None


def get_attributes(obj, attr_name, ddp_on):
    functions = []
    func = get_attribute(obj, attr_name, ddp_on)
    if func is not None:
        functions.append(func)
    classifier = get_attribute(obj, 'classifier', ddp_on)
    if classifier is not None:
        func = get_attribute(classifier, attr_name)
        if func is not None:
            functions.append(func)
    criterion = get_attribute(obj, 'criterion', ddp_on)
    if criterion is not None:
        func = get_attribute(criterion, attr_name)
        if func is not None:
            functions.append(func)
    return functions


def print_class_mean_idx(class_means, logger, gpu_rank=0):
    nums = []
    logger.info("Class mean code:", gpu_rank)
    for class_mean in class_means:
        code = (class_mean == 1).nonzero()
        code = code.detach().cpu().numpy().reshape(-1)
        logger.info(code, gpu_rank)
        nums.append(len(code))
    logger.info("Number of ones in each class code:", gpu_rank)
    logger.info(nums, gpu_rank)


def check_nan_array_numpy(arr):
    if np.any(np.isnan(arr)):
        arr = np.nan_to_num(arr)
    return arr


def check_nan_array_torch(arr):
    if torch.any(torch.isnan(arr)):
        arr = torch.nan_to_num(arr)
    return arr


def store_cifar_data(data_dir, store_dir):
    # Unpickle function provided by the CIFAR hosts
    images, labels = [], []
    # for batch in data_dir.glob("data_batch_*"):
    for batch in data_dir.glob("test_batch"):
        batch_data = unpickle(batch)
        for i, flat_im in enumerate(batch_data[b"data"]):
            im_channels = []
            # Each image is flattened, with channels in order of R, G, B
            for j in range(3):
                im_channels.append(
                    flat_im[j * 1024: (j + 1) * 1024].reshape((32, 32))
                )
            # Reconstruct the original image
            images.append(np.dstack(im_channels))
            # Save the label
            labels.append(batch_data[b"labels"][i])

    print("Loaded CIFAR-10 training set:")
    print(f" - np.shape(images)     {np.shape(images)}")
    print(f" - np.shape(labels)     {np.shape(labels)}")

    store_many_disk(images, labels, store_dir)


def store_mnist_data(raw_dir, store_dir, train):
    dataset = torchvision.datasets.MNIST(root=raw_dir, train=train, download=True)
    store_many_disk(dataset.data.numpy(), dataset.targets.numpy(), store_dir)


def store_many_disk(images, labels, store_path):
    """ Stores an array of images to disk
        Parameters:
        ---------------
        images:       images array, (N, 32, 32, 3) to be stored
        labels:       labels array, (N, 1) to be stored
    """
    # Save all the images one by one
    for i, image in enumerate(images):
        Image.fromarray(image).save(store_path[labels[i]] / f"{i}.png")


def get_samples(paths, transform):
    samples = []
    simpler_transform = torchvision.transforms.Compose(transform.transforms[:-1])
    normalize_transform = transform.transforms[-1]
    from PIL import Image
    for (path, label, show_label) in paths:
        with open(path, 'rb') as f:
            image = Image.open(f).convert('RGB')
            image_tensor = simpler_transform(image)

            processed_image = image_tensor.numpy()
            processed_image = np.transpose(processed_image, (1, 2, 0))

            image_tensor = normalize_transform(image_tensor)
            image_tensor = image_tensor.unsqueeze(0)
            image_tensor.requires_grad = True
            samples.append(
                {'input_image': processed_image,
                 'orig_img': image,
                 'image_tensor': image_tensor,
                 'target_class': label,
                 'show_class': show_label}
            )

    return samples


def prepare_hierarchical_data_inaturalist18():
    directory = '/Users/yihang/Datasets/iNaturalist2018/train_val2018'

    coarse_labels = get_immediate_sub_folders(directory)
    coarse_labels = sorted(coarse_labels)
    coarse2fine_dict = {}
    fine2coarse = np.ones(8142, dtype=int) * -1
    coarse_labels_dict = {}

    for coarse_label, p in enumerate(coarse_labels):
        coarse_labels_dict[p] = coarse_label
        folders = f"{directory}/{p}"
        folders = get_immediate_sub_folders(folders)
        fine_labels_subset = [int(i) for i in folders]
        coarse2fine_dict[coarse_label] = fine_labels_subset

        for i in fine_labels_subset:
            assert np.equal(fine2coarse[i], -1), f"class {i} is assigned: {fine2coarse[i]}."
            fine2coarse[i] = coarse_label

    save_file = {
        "coarse_labels": coarse_labels_dict,
        "coarse2fine": coarse2fine_dict,
        "fine2coarse": fine2coarse
    }

    pickle_file('iNaturalist2018_hierarchical.pkl', save_file)


def prepare_hierarchical_data_cifar100():
    coarse2fine_string_dict = {
        'aquatic_mammals': ['beaver', 'dolphin', 'otter', 'seal', 'whale'],
        'fish': ['aquarium_fish', 'flatfish', 'ray', 'shark', 'trout'],
        'flowers': ['orchid', 'poppy', 'rose', 'sunflower', 'tulip'],
        'food_containers': ['bottle', 'bowl', 'can', 'cup', 'plate'],
        'fruit_and_vegetables': ['apple', 'mushroom', 'orange', 'pear', 'sweet_pepper'],
        'household_electrical_devices': ['clock', 'keyboard', 'lamp', 'telephone', 'television'],
        'household_furniture': ['bed', 'chair', 'couch', 'table', 'wardrobe'],
        'insects': ['bee', 'beetle', 'butterfly', 'caterpillar', 'cockroach'],
        'large_carnivores': ['bear', 'leopard', 'lion', 'tiger', 'wolf'],
        'large_man-made_outdoor_things': ['bridge', 'castle', 'house', 'road', 'skyscraper'],
        'large_natural_outdoor_scenes': ['cloud', 'forest', 'mountain', 'plain', 'sea'],
        'large_omnivores_and_herbivores': ['camel', 'cattle', 'chimpanzee', 'elephant', 'kangaroo'],
        'medium_mammals': ['fox', 'porcupine', 'possum', 'raccoon', 'skunk'],
        'non-insect_invertebrates': ['crab', 'lobster', 'snail', 'spider', 'worm'],
        'people': ['baby', 'boy', 'girl', 'man', 'woman'],
        'reptiles': ['crocodile', 'dinosaur', 'lizard', 'snake', 'turtle'],
        'small_mammals': ['hamster', 'mouse', 'rabbit', 'shrew', 'squirrel'],
        'trees': ['maple_tree', 'oak_tree', 'palm_tree', 'pine_tree', 'willow_tree'],
        'vehicles_1': ['bicycle', 'bus', 'motorcycle', 'pickup_truck', 'train'],
        'vehicles_2': ['lawn_mower', 'rocket', 'streetcar', 'tank', 'tractor'],
    }

    meta = unpickle(f"./saved/data/cifar/cifar-100-python/meta", encoding="latin1")

    coarse_labels_dict = {_class: i for i, _class in enumerate(meta['coarse_label_names'])}
    fine_labels_idx = {_class: i for i, _class in enumerate(meta['fine_label_names'])}
    coarse2fine = [[] for _ in coarse2fine_string_dict.keys()]
    fine2coarse = np.ones(100, dtype=int) * -1
    for coarse_label, idx in coarse_labels_dict.items():
        fine_labels_subset = coarse2fine_string_dict[coarse_label]
        coarse2fine[idx] = []
        for fine_label in fine_labels_subset:
            fine_label_idx = fine_labels_idx[fine_label]
            fine2coarse[fine_label_idx] = idx
            coarse2fine[idx].append(fine_label_idx)

    save_file = {
        "coarse_labels": meta['coarse_label_names'],
        "fine_labels": meta['fine_label_names'],
        "coarse2fine": coarse2fine,
        "fine2coarse": fine2coarse
    }
    pickle_file('cifar100_hierarchical.pkl', save_file)


def get_num_data_per_class(labels, num_classes):
    _labels = np.array(labels)
    sample_id = np.array(range(len(labels)))
    if num_classes is None:
        return None, None, sample_id
    # the mask only works for array not for list.
    class_ids = sorted(list(set(labels)))
    assert len(class_ids) == num_classes, len(set(labels))
    sample_id_per_class = [sample_id[_labels == i] for i in class_ids]
    num_per_class = np.array([len(sample_id_per_class[i]) for i in range(num_classes)], dtype=float)
    return num_per_class, sample_id_per_class, sample_id


def change_to_coarse_labels(fine2coarse_labels, train_targets=None, test_targets=None):
    if train_targets is not None:
        new_train_targets = []
        for target in train_targets:
            new_train_targets.append(int(fine2coarse_labels[target]))
        train_targets = new_train_targets
    if test_targets is not None:
        new_test_targets = []
        for target in test_targets:
            new_test_targets.append(int(fine2coarse_labels[target]))
        test_targets = new_test_targets
    return train_targets, test_targets


def change_to_new_fine_labels(original_targets, original_samples, coarse_label_id, fine2coarse_labels,
                              orig_to_new_labels):
    mask = torch.zeros(len(original_targets))
    for _i, _target in enumerate(original_targets):
        if fine2coarse_labels[_target] == coarse_label_id:
            mask[_i] = 1
    selected_samples = original_samples[mask == 1]
    new_targets = []
    for _i, _target in enumerate(original_targets):
        if mask[_i] == 1:
            new_targets.append(orig_to_new_labels[_target])
    assert len(new_targets) == len(selected_samples)
    return new_targets, selected_samples


def calculate_bias_term_proportion(saved_values=None, learning_model=None, dataset=None, root=None, ddp_on=False):
    if 'logits' in saved_values:  # to calculate the percentage of bias term to the logits
        classifier = get_attribute(learning_model, 'classifier', ddp_on=ddp_on)
        fc = get_attribute(classifier, 'fc')
        bias_term = get_attribute(fc, 'bias').detach().cpu().numpy()

        logits_mean = np.array(saved_values['logits']).mean(axis=1)
        proportions = np.divide(bias_term, logits_mean)

        # if dataset is not None:
        sorted_id = sorted([*range(dataset.num_classes)], key=lambda c: dataset.train_num_per_cls_dict[c], reverse=True)
        proportions = proportions[sorted_id]
        torch.save(proportions, root / 'bias_term.pkl')


@torch.no_grad()
def gather_all(tensor):
    if dist.is_initialized():
        tensors_gather = [torch.ones_like(tensor) for _ in range(dist.get_world_size())]
        dist.all_gather(tensors_gather, tensor, async_op=False)
        return tensors_gather
    else:
        return None


def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    if isinstance(tensor, list):
        tensor = torch.stack(tensor, dim=0)
    output = gather_all(tensor)
    if output is None:
        return tensor
    else:
        output = torch.cat(output, dim=0)
        return output


def reduce_all(tensor, form='SUM', logger=None):
    dist.all_reduce(tensor, op=getattr(dist.ReduceOp, form))
    logger.info("Results are reduced across GPUs")
    return tensor


# if __name__ == '__main__':
#     mode = 'test'
#     disk_dir = Path("/Users/yihang/Datasets/mnist/" + mode)
#     disk_dirs = []
#     for i in range(10):
#         specific_path = disk_dir / f'{i}'
#         specific_path.mkdir(parents=True, exist_ok=True)
#         disk_dirs.append(specific_path)
#     # Path to the unzipped CIFAR data
#     store_mnist_data(raw_dir=Path("saved/data/mnist/"), store_dir=disk_dirs, train=mode == 'train')


def summarise_results(parent_dir, sort, prefix=1, class_mean='Class_mean'):
    sub_folders = get_immediate_sub_folders(parent_dir)
    keys = {'train': ['Train', 'epoch[', ': [recall]  acc@head:'],
            'test': ['Eval', 'epoch[', ': [recall]  acc@head:']
            }
    if class_mean is not None:
        keys[class_mean] = [f"{class_mean.replace('_', ' ')}:	weight-sum mean: @head:"]

    results = {}
    for sub_folder in sub_folders:
        results[str(sub_folder)] = {}
        try:
            f = open(os.path.join(parent_dir, sub_folder, 'log/log_file.txt_0.txt'), "r")
        except FileNotFoundError:
            f = open(os.path.join(parent_dir, sub_folder, 'log_file.txt'), "r")
        lines = f.readlines()
        for i in range(1, len(lines)):
            line = lines[-i]
            for key, values in keys.items():
                recognised = True
                for value in values:
                    if value not in line:
                        recognised = False
                        break
                if recognised:
                    results[str(sub_folder)][key] = line
                    break

            if len(results[str(sub_folder)]) == len(keys):
                break
        if len(results[str(sub_folder)]) != len(keys):
            assert False, sub_folder
        f.close()

    names = [['cifar10_', f'_{prefix}_0_'], ['cifar10_', f'_{prefix}_0-1_'],
             ['cifar10_', f'_{prefix}_0-02_'], ['cifar10_', f'_{prefix}_0-01_'],
             ['cifar100_', f'_{prefix}_0_'], ['cifar100_', f'_{prefix}_0-1_'],
             ['cifar100_', f'_{prefix}_0-02_'], ['cifar100_', f'_{prefix}_0-01_'],
             ['mnist_', f'_{prefix}_0_'], ['mnist_', f'_{prefix}_0-1_'],
             ['mnist_', f'_{prefix}_0-02_'], ['mnist_', f'_{prefix}_0-01_'],
             [f'imagenet_1k_{prefix}', ''],
             [f'imagenet_lt_{prefix}', ''],
             [f'iNaturalist2018_{prefix}', '']
             ]

    with open(f'{parent_dir}/result', 'w') as f:
        for key, _ in keys.items():
            if key in results[str(sub_folders[0])]:
                f.write(f'\n\n{key}:\n')
                recorded = []
                if sort:
                    for name in names:
                        for sub_folder in sub_folders:
                            if name[0] in sub_folder and name[1] in sub_folder:
                                f.write(f"{sub_folder}\t{results[str(sub_folder)][key]}")
                                recorded.append(sub_folder)
                                break

                for sub_folder in sub_folders:
                    if sub_folder not in recorded:
                        f.write(f"{sub_folder}\t{results[str(sub_folder)][key]}")

    f.close()


def read_backgrounds(parent_path, size=(28, 28)):
    transform = transforms.Compose([transforms.Resize(size)])
    backgrounds = []

    def get_background(image_path):
        background = Image.open(image_path).convert('RGB')
        background = transform(background)
        backgrounds.append(np.asarray(background))

    files = get_immediate_files(parent_path)
    try:
        files.remove('.DS_Store')
    except ValueError:
        pass
    if len(files) == 0:
        sub_folders = get_immediate_sub_folders(parent_path)
        for sub_folder in sub_folders:
            files = get_immediate_files(f"{parent_path}/{sub_folder}")
            for file in files:
                get_background(f"{parent_path}/{sub_folder}/{file}")

    else:
        for file in files:
            get_background(f"{parent_path}/{file}")

    return backgrounds


def validate_checkpoints(parent_dir="saved/train/temp_store"):
    sub_folders = get_immediate_sub_folders(parent_dir)
    for sub_folder in tqdm(sub_folders):
        full_path = f"{parent_dir}/{sub_folder}/ckps/last_ckp.tar"
        try:
            torch.load(full_path, map_location=torch.device('cpu'))
        except Exception as e:
            print(e, full_path)
    print("Validation completed.")

def kl_divergence(P_ideal, Q_test):
    if isinstance(P_ideal, torch.Tensor):
        P_ideal = P_ideal.detach().cpu().numpy()
    if isinstance(Q_test, torch.Tensor):
        Q_test = Q_test.detach().cpu().numpy()

    assert isinstance(P_ideal, np.ndarray)
    assert isinstance(Q_test, np.ndarray)
    # Avoid division by zero and log(0) by replacing zero values with a small number
    epsilon = 1e-10
    P_ideal = np.clip(P_ideal, epsilon, 1)
    Q_test = np.clip(Q_test, epsilon, 1)

    return np.sum(P_ideal * np.log(P_ideal / Q_test))


def quantify_model_preference(dataset, method):
    from datasets.checkpoints import get_checkpoint
    from utils.meter import get_group_mean

    train_acc = \
        torch.load(get_checkpoint(data_name=dataset.key, method=method, _type='base') + "/train_class_values.pk",
                   map_location=torch.device('cpu'), weights_only=False)['recall']
    test_acc = torch.load(get_checkpoint(data_name=dataset.key, method=method, _type='base') + "/test_class_values.pk",
                          map_location=torch.device('cpu'), weights_only=False)['recall']

    train_acc *= 100
    test_acc *= 100

    train_mean, _ = get_group_mean(dataset, train_acc, measure='mean')

    test_acc_mean, _ = get_group_mean(dataset, test_acc, measure='mean')

    diff = (test_acc - train_acc) / train_acc * 100

    diff_mean, _ = get_group_mean(dataset, diff, measure='mean')

    preference = diff + train_acc

    preference_mean, _ = get_group_mean(dataset, preference, measure='mean')

    preference_sd = torch.std(preference, dim=0)

    print(
        f"{dataset.key} {method} bias: {train_mean}, variance: {diff_mean}, preference: {preference_mean}, preference_sd: {preference_sd}")


def check_identical(manually, ground_truth, name=""):
    # We use a slightly looser tolerance because float32 accumulators drift slightly
    is_correct = torch.allclose(manually, ground_truth, atol=1e-5)
    diff = (manually - ground_truth).abs().max().item()
    print(f"Max Absolute {name}Difference: {diff:.8f}")
    print(f"Derivation Verified: {is_correct}")
    assert is_correct


def gini_coefficient(values):
    if isinstance(values, torch.Tensor):
        values = values.detach().cpu().numpy()
    # Ensure the array is sorted
    values = np.sort(values)
    # Number of values
    n = len(values)
    # Calculate Gini coefficient using the formula
    index = np.arange(1, n + 1)
    gini = (np.sum((2 * index - n - 1) * values)) / (n * np.sum(values))
    return gini


if __name__ == '__main__':
    print(os.getcwd())
    validate_checkpoints("saved/results_for_report/bcl/bscl")
    # convert_zip2tar(input_zip="/Users/yihang/Downloads/archive.zip", output_path="/Users/yihang/Downloads/", max_count=600000)
