import os
import time
from pathlib import Path

import numpy as np
import torch
from torch.distributed import init_process_group, destroy_process_group

import models
from utils.utils import (load_yaml, Logging, set_reproducibility, get_free_port,
                         MultiEpochsDataLoader, count_parameters)


def _get_str_config(cfg):
    string = ""
    return __get_str_config(cfg, string, horizon="", first=True)


def __get_str_config(item, string, horizon, first=False):
    if not first:
        horizon += "\t"
        string += '\n'
    for k, v in item.items():
        string += f"{horizon}{k}:"
        if isinstance(v, dict):
            string = __get_str_config(v, string, horizon)
        else:
            string += f' {v}\n'
    return string


def init_obj(obj, module, config, *args, allow_override=False, **kwargs):
    """
    Finds a function handle with the name given as 'type' in config, and returns the
    instance initialized with corresponding arguments given.

    `object = config.init_obj('name', module, a, b=1)`
    is equivalent to
    `object = module.name(a, b=1)`
    """
    module_name = config[obj]['name']
    module_args = dict(config[obj]['args']) if 'args' in config[obj] else dict()
    if not allow_override:
        assert all([k not in module_args for k in kwargs]), 'Overwriting kwargs given in config file is not allowed'
    module_args.update(kwargs)
    module = getattr(module, module_name)(*args, **module_args)
    return module


def create_learning_model(config, logger, device=None, gpu_id=None, gpu_rank=None, **kwargs):
    logger.info(f"==============Learning model==============", gpu_rank)

    module_args = config['learning_model']['args']
    if 'classifier' in config:
        module_args['classifier'] = config['classifier']
    if 'loss' in config:
        module_args['loss'] = config['loss']
    if 'backbone' in config:
        module_args['backbone'] = config['backbone']

    module_args.update(kwargs)
    module = getattr(models, config['learning_model']['name'])(**module_args)

    if hasattr(module, 'representation_model'):
        if 'fix_representation' in config['trainer']:
            module.representation_model.requires_grad_(False)
            # for param in representation_model.parameters():
            #     param.requires_grad = False
            logger.info('**The representation model is fixed, not trainable.', gpu_rank)

        logger.info(
            f"Backbone total parameters: "
            f"{count_parameters(module.representation_model, only_trainable=True)}", gpu_rank)
        logger.info(
            f"Backbone trainable parameters: "
            f"{count_parameters(module.representation_model, only_trainable=True)}", gpu_rank)
    logger.info(f"Total parameters: {count_parameters(module)}", gpu_rank)
    logger.info(f"Total trainable parameters: {count_parameters(module, only_trainable=True)}",
                gpu_rank)

    module.to(device)
    if config['ddp']['on']:
        module = torch.nn.parallel.DistributedDataParallel(module, device_ids=[gpu_id])
    return module


def _setup_ddp(config, args):
    num_unused_gpus = args.nodes * args.gpus - args.world_size
    if args.world_size > 0 and num_unused_gpus >= args.gpus:
        raise Exception(f"Too many nodes are booked: "
                        f"{args.nodes} nodes * {args.gpus} gpus >> {args.world_size} world_size.")

    ddp = config['ddp']
    # =================setup hardware environment
    # assert args.gpus == torch.cuda.device_count()
    # the total number of GPUs to run considering all the applied nodes,
    # assuming each node has the same number of GPUs
    ddp['world_size'] = args.world_size
    ddp['num_nodes'] = args.nodes
    ddp['num_gpus_per_node'] = [args.gpus for _ in range(args.nodes)]
    ddp['num_gpus_per_node'][-1] -= num_unused_gpus

    if 'SLURM_PROCID' in os.environ:
        ddp['node_rank'] = int(os.environ['SLURM_PROCID'])
    else:
        ddp['node_rank'] = args.nr

    # the batch will be divided by all the available GPUs
    if ddp['world_size'] > 1:
        ddp['on'] = True
        if config['dataset']['args']['batch_size'] is None:
            config['dataset']['args']['batch_size'] = int(
                int(config['dataset']['args']['total_batch_size']) / ddp['world_size']
            )
        if args.ip is not None:
            if args.port is None:
                args.port = get_free_port()
            ddp['dist_url'] = f'tcp://{args.ip}:{args.port}'
        else:
            if 'MASTER_ADDR' not in os.environ:
                raise Exception('Master address is not set.')
            if 'MASTER_PORT' not in os.environ:
                raise Exception('Master port is not set.')
    else:
        if config['dataset']['args']['batch_size'] is None:
            config['dataset']['args']['batch_size'] = int(config['dataset']['args']['total_batch_size'])


def init_ddp(config, logger, gpu_id):
    ddp = config['ddp']
    data_loader_func = MultiEpochsDataLoader
    if ddp['on']:  # initialize torch distributed parallel package for running with multiple nodes.
        # This is the global rank of the process within all the processes (one process per GPU)
        # gpus_ranks includes the rank for all the gpus
        count = 0
        gpu_ranks = [[0 for _ in range(ddp['num_gpus_per_node'][node_id])] for node_id in range(ddp['num_nodes'])]
        for node_id in range(ddp['num_nodes']):
            for g_id in range(ddp['num_gpus_per_node'][node_id]):
                gpu_ranks[node_id][g_id] = count
                count += 1

        gpu_rank = gpu_ranks[ddp['node_rank']][gpu_id]
        logger.info(f"Ranks for all the applied GPUs: {gpu_ranks}", gpu_rank=gpu_rank)
        logger.info(f"Initializing distributed package for GPU {gpu_rank}/{ddp['world_size']} "
                    f"(node {ddp['node_rank']}, GPU {gpu_id}/{ddp['num_gpus_per_node'][ddp['node_rank']]}) "
                    f"with {ddp['dist_url']}.....",
                    gpu_rank=gpu_rank)  # print out this for all GPUs
        init_process_group(backend=ddp['dist_backend'], init_method=ddp['dist_url'],
                           world_size=ddp['world_size'], rank=gpu_rank)
        logger.info(f"Distributed package is initialized for GPU {gpu_rank}/{ddp['world_size']}", gpu_rank=gpu_rank)

        device = f'cuda:{gpu_id}'
        torch.cuda.set_device(gpu_id)

        distributed_sampler = {
            'func': torch.utils.data.distributed.DistributedSampler,
            'args':
                {
                    'num_replicas': config['ddp']['world_size'],
                    'rank': gpu_rank,
                    'seed': config['trainer']['seed']
                }
        }

        config['dataset']['args']['distributed_sampler'] = distributed_sampler
        logger.info('===> Distributed sampler is ready', gpu_rank=gpu_rank)
    else:

        if torch.cuda.is_available():
            gpu_rank = gpu_id
            device = f'cuda:0'
            logger.info('===> using one GPU for training', gpu_rank=gpu_rank)
        else:
            gpu_rank = 0  # for convenient purpose.
            device = 'cpu'
            logger.info('===> using CPU', gpu_rank=gpu_rank)
            data_loader_func = torch.utils.data.DataLoader

    if config['trainer']['mode'] not in ['train', 'test']:
        data_loader_func = torch.utils.data.DataLoader
    return data_loader_func, device, gpu_rank


def _check_args(args):
    if args.nodes == 1:
        args.nr = 0

    if not args.deterministic:
        args.seed = np.random.randint(10000000)

    if not torch.cuda.is_available():
        args.print_freq = 1
    return args


def prepare_dirs(config, time_str=""):
    dirs = dict()
    if not torch.cuda.is_available():
        saved_path_parent = '/Users/yihang/cache/'
    else:
        saved_path_parent = 'saved/'
    dirs['save_path'] = Path(
        f"{saved_path_parent}"
        f"{config['trainer']['mode']}/"
        f"{time_str}{config['trainer']['name']}_gpus{config['ddp']['world_size']}_seed{config['trainer']['seed']}"
    )
    if config['trainer']['mode'] in ['test', 'train'] and config['trainer']['model_update']:
        dirs['model_dir'] = dirs['save_path'] / Path('ckps')
        dirs['model_dir'].mkdir(parents=True, exist_ok=True)
    return dirs


def _clear_config(config):
    # clean up the unused taps in the template
    for k in list(config.keys()):
        if isinstance(config[k], dict):
            _clear_config(config[k])
        else:
            if config[k] is None:
                del config[k]


def __load_config(template, config, msg_no_key):
    for k, v in config.items():
        assert k in template.keys(), f'{msg_no_key}: {k}'
        if isinstance(config[k], dict):
            __load_config(template[k], config[k], msg_no_key)
        else:
            template[k] = config[k]
    return template


def _load_config(existing_config, cfg_file):
    msg_no_key = 'key is not found in the template file'
    config = load_yaml(cfg_file)
    existing_config = __load_config(existing_config, config, msg_no_key)
    return existing_config


def _get_common_config(cfg_file):
    if isinstance(cfg_file, str):
        print(cfg_file)
        cfg_file = eval(cfg_file)
    assert isinstance(cfg_file, list)
    common_cfg = {
        'cifar': 'config/cifar/common_cifar.yaml',
        'mnist': 'config/mnist/common_mnist.yaml',
        'imagenet_lt': 'config/imagenet_lt/common_imagenet_lt.yaml',
        'imagenet_1k': 'config/imagenet_1k/common_imagenet_1k.yaml',
        'place': 'config/place_lt/common_place_lt.yaml',
        'iNaturalist': 'config/iNaturalist2018/common_iNaturalist2018.yaml',
        'lvis': 'config/lvis/lvis_v1_common.yaml'
    }

    cfg_common = None
    for key, value in common_cfg.items():
        if key in cfg_file[0]:
            cfg_common = value
            return cfg_common, cfg_file
    assert cfg_common is not None


def load_cfg_file(cfg_file, clean=True, args=None):
    config = load_yaml('utils/menu.yaml')

    if args is not None:
        config['trainer']['task_worker'] = args.task_worker
        config['trainer']['mode'] = args.mode
        config['trainer']['resume'] = bool(args.resume)
        config['trainer']['checkpoint'] = None if args.checkpoint == "None" else args.checkpoint

        config['trainer']['model_update'] = bool(args.model_update)
        config['trainer']['start_epoch'] = int(args.start_epoch)
        config['trainer']['num_epochs'] = int(args.num_epochs) if config['trainer']['model_update'] else 1
        config['trainer']['run_epochs'] = eval(args.run_epochs) if isinstance(
            args.run_epochs, str) else args.run_epochs
        config['trainer']['run_test'] = bool(args.run_test)
        config['trainer']['final_test'] = args.final_test
        config['trainer']['save_all_ckp'] = args.save_all_ckp
        config['trainer']['record_more_details'] = eval(args.record_more_details) if isinstance(
            args.record_more_details, str) else args.record_more_details
        config['dataset']['args']['hpc'] = args.hpc
        config['trainer']['seed'] = int(args.seed)
        config['trainer']['print_freq'] = int(args.print_freq)
        config['dataset']['args']['num_workers'] = int(args.wks)
        config['trainer']['classifier_rate'] = args.classifier_rate
        config['trainer']['fp16'] = args.fp16
        if bool(args.test_checkpoint):
            config['trainer']['run_epochs'] = 1
            config['trainer']['model_update'] = False
            assert config['trainer']['checkpoint'] is not None, "Please specify a checkpoint file."

    cfg_common, cfg_file = _get_common_config(cfg_file)
    config = _load_config(config, cfg_common)

    log_file_name = ""
    for cfg in cfg_file:
        name = cfg
        if ".yaml" in cfg:
            config = _load_config(config, cfg)
            name = name.split(".")[0]
            name = name.split("/")[-1]
        log_file_name += f"_{name}"
    config['trainer']['name'] = log_file_name

    if args is not None and args.opt is not None:
        opts = args.opt.split('&')
        for opt in opts:
            opt = opt.split('@')
            if "str" in opt[-1]:
                opt[-1] = opt[-1].replace("(", "('").replace(")", "')")
            else:
                try:
                    opt[-1] = eval(opt[-1])
                except NameError as _:
                    pass
            if len(opt) == 4:
                config[opt[0]][opt[1]][opt[2]] = opt[-1]
            elif len(opt) == 3:
                config[opt[0]][opt[1]] = opt[-1]
            else:
                raise Exception("Tunings are not set correctly.")
            config['trainer']['name'] += f"_{opt[-1]}".replace(".", "-").replace(",", "-")

    if config['trainer']['run_epochs'] is None:
        config['trainer']['run_epochs'] = config['trainer']['num_epochs']
    if config['trainer']['resume']:
        assert config['trainer']['checkpoint'] is not None, 'Please setup path for loading checkpoints.'

    if args is not None:
        _setup_ddp(config, args)

    # =================setup folders===========
    # time_identify
    if args is None or args.time is None:
        # this only valid when running on a single machine
        if config["ddp"]['world_size'] > 1:
            raise Exception("args.time is not provided.")
        time_str = time.strftime('%Y-%m-%d_%H-%M-%S')
    else:
        time_str = args.time
    dirs = prepare_dirs(config, time_str)
    # ============ setup logger
    # when running with multiprocessing, only log for process 0 of node args.nr
    logger = Logging(dirs['save_path'] / "log", 'log_file')

    if clean:
        _clear_config(config)

    logger.info(f"Folder created: {dirs['save_path']}", gpu_rank=config['ddp']['node_rank'])
    logger.info(_get_str_config(config), gpu_rank=config['ddp']['node_rank'])

    gpus_per_node = torch.cuda.device_count()
    logger.info(f"Node {config['ddp']['node_rank']} has {gpus_per_node} GPUs", gpu_rank=config['ddp']['node_rank'])

    del config['dataset']['args']['total_batch_size']
    return config, dirs, logger


class ConfigParser:
    def __init__(self, args):
        args = _check_args(args)
        set_reproducibility(args.seed)
        self._config, self._dirs, self.logger = load_cfg_file(cfg_file=args.cfg, args=args)

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        _get_str_config(self._config)

    # setting read-only attributes
    @property
    def config(self):
        return self._config

    @property
    def dirs(self):
        return self._dirs

    def stop_ddp(self):
        if self.config['ddp']['on']:
            destroy_process_group()

    def output(self, gpu_rank):
        """only output at rank 0 GPU out of all the available GPUs"""
        return not self.config['ddp']['on'] or (self.config['ddp']['on'] and gpu_rank == 0)

    def adapt_config_file(self, gpu_rank):
        if 'metric' in self._config['trainer']['name']:
            self._config['classifier']['args']['store_path'] = self.dirs['save_path']
            if gpu_rank is not None:
                self._config['classifier']['args']['gpu_rank'] = gpu_rank
