import argparse
import time

import torch
import torch.multiprocessing as mp

import workers
from utils import *
from utils.configurate import init_ddp


def main(args):
    config = ConfigParser(args)
    task_worker = getattr(workers, config.config['trainer']['task_worker'])
    if config.config['ddp']['on']:
        # Use torch.multiprocessing.spawn to launch distributed processes:
        # the main_worker process function
        num_gpus = config.config['ddp']['num_gpus_per_node'][config.config['ddp']['node_rank']]
        mp.spawn(main_worker, nprocs=num_gpus, args=(config, task_worker,))
    else:
        # Simply call the main_worker function
        main_worker(0 if torch.cuda.is_available() else None, config, task_worker)


def main_worker(gpu_id, config, task_worker):
    """gpu_id is the id per node, while gpu_rank is the gpu rank among all the available gpus in the nodes"""
    try:
        logger = config.logger
        start_time = time.time()
        cfg = config.config

        data_loader_func, device, gpu_rank = init_ddp(cfg, logger, gpu_id)
        config.adapt_config_file(gpu_rank)

        task_worker(config, data_loader_func, device, gpu_rank, gpu_id, start_time)

        logger.info(f"Completed, time: {round(time.time() - start_time)}s", gpu_rank=gpu_rank)
    finally:
        config.stop_ddp()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='training')
    # ============= for trainer
    parser.add_argument('--task_worker', type=str, default=('bbox_worker', 'classification_worker')[1],
                        help='the worker to use for running the ddp')
    parser.add_argument('--cfg',
                        default=['config/cifar/cifar10LT10_cbs_p.yaml'],
                        type=str,
                        help='experiment configure file name')
    parser.add_argument('--opt', type=str,
                        # default="classifier@args@ibs@1",
                        help='for tuning contain parameters using config file param_tuning.yaml'
                             'e.g., classifier@args@enforced_groups@list([1,4])&'
                             'classifier@args@enforced_groups@list([1,4])'
                        )
    parser.add_argument('--mode', default=('train', 'validate', 'visualize', 'rest')[0], type=str)
    parser.add_argument('--resume', default=0, type=int,
                        help='whether resume training from a checkpoint.')
    parser.add_argument('--test_checkpoint', default=False, type=int,
                        help='whether test checkpoint, true will disable model update '
                             'and run one epoch on the training and testing data')
    parser.add_argument('--checkpoint', default="None", type=str,
                        help='the path for the checkpoint to be loaded or True')
    parser.add_argument('--model_update', default=1, type=int,
                        help='True if update model parameters')
    parser.add_argument('--start_epoch', default=0, type=int,
                        help='number of the epoch to start for running.')
    parser.add_argument('--run_epochs', default=None, type=str,
                        help='actually number of epochs to run')
    parser.add_argument('--num_epochs', default=200, type=int,
                        help='number of epochs for training')
    parser.add_argument('--final_test', default=1, type=int,)
    parser.add_argument('--run_test', default=0, type=int,
                        help='whether or not run test during training')
    parser.add_argument('--save_all_ckp',
                        # default={'step': 80, 'epochs': 1},
                        help='whether to save all the checkpoints during model training')
    parser.add_argument('--record_more_details',
                        default=[], type=str,
                        help="options: 'ce_grads', 'scl_grads', 'class_prob', 'class_values', 'target_prob'")

    parser.add_argument('--hpc', default=None, type=str,
                        help="use time to create a unique folder for recording the experiment")
    parser.add_argument('--deterministic', default=True, type=bool,
                        help='fix random seed?')
    parser.add_argument('--seed', default=1, type=int,
                        help='random seed')
    parser.add_argument('--classifier_rate', default=0.1, type=float,
                        help='The rate for important or negligible values')

    # ================ for logging
    parser.add_argument('--print_freq', default=40, type=int,
                        help='the frequency of recording the training log.')
    parser.add_argument('--time', default=None, type=str,
                        help="use time to create a unique folder for recording the experiment")
    # ============= for ddp
    parser.add_argument('--nodes', default=1, type=int, metavar='N',
                        help='the total number of nodes we’re going to use')
    parser.add_argument('--gpus', default=0, type=int,
                        help='the total number of gpus available on each node')
    parser.add_argument('--world_size', default=0, type=int,
                        help='the total number of gpus we need to run the experiment, '
                             'e.g., node 0 2 gpus, node 1 3 gpus, world_size=5')
    parser.add_argument('--nr', default=None, type=int,
                        help='the rank of the current node within all the nodes, and goes from 0 to args.nodes-1')
    parser.add_argument('--ip', default=None, help='the ip address for MASTER_ADDR')
    parser.add_argument('--port', default=None, help='the free port, set arbitrarily as long as it is free', )
    parser.add_argument('--wks', default=2, type=int,
                        help='number of workers for each GPU, this ideally should be around 15, '
                             'too large or too small will make the system inefficient'
                             'https://chtalhaanwar.medium.com/pytorch-num-workers-a-tip-for-speedy-training'
                             '-ed127d825db7#:~:text=Theoretically%2C%20greater%20the%20num_workers%2C%20more,'
                             'performance%20start%20diminishing%20beyond%20that.')

    # fp16
    parser.add_argument('--fp16', default=False, help=' fp16 training')

    main(parser.parse_args())
