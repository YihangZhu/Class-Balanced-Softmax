import torch
import torch.distributed as dist
import torch.optim
import torch.utils.data

import datasets
from utils import *
from utils.configurate import init_obj, create_learning_model


def classification_worker(config, data_loader_func, device, gpu_rank, gpu_id, start_time):
    """gpu_id is the id per node, while gpu_rank is the gpu rank among all the available gpus in the nodes"""
    logger = config.logger
    cfg = config.config
    config.adapt_config_file(gpu_rank)
    if cfg['trainer']['model_update']:
        lr_scheduler = init_obj('lr_scheduler', lr_schedulers, cfg,
                                lr=cfg['optimizer']['args']['lr'],
                                num_epochs=cfg['trainer']['num_epochs']
                                )
        logger.info(
            f"==============Learning rate scheduler==============\ninit lr: {lr_scheduler.lr}", gpu_rank)
    else:
        lr_scheduler = None



    train_dataset = init_obj('dataset', datasets, cfg, logger=logger, data_loader_func=data_loader_func,
                             gpu_rank=gpu_rank)
    test_dataset = None if not hasattr(train_dataset, 'test_dataloader') else train_dataset

    # with torch.no_grad():
    #     # feature_model.linear.weight.copy_(classifier_model.fc.weight)
    #     classifier_model.fc.weight.copy_(feature_model.linear.weight)
    #     classifier_model.fc.bias.copy_(feature_model.linear.bias)
    # criterion = config.init_obj('loss', models, device=device, dataset=dataset)
    learning_model = create_learning_model(cfg, logger, device=device, gpu_id=gpu_id, gpu_rank=gpu_rank,
                                           dataset=train_dataset)
    # print_class_mean_idx(learning_model.classifier.class_mean_code, logger, gpu_rank)
    optimizer = init_obj('optimizer', torch.optim, cfg, learning_model.parameters()
                         )
    # check_linear_classifier(learning_model, dataset, logger, dist.is_initialized(), 0)

    # lr_scheduler = LR_scheduler(optimizer, name=cfg['lr_scheduler']['name'])
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, mode='min', patience=5)
    # reduce learning rate when error plateau occurs.
    # weight = torch.ones(dataset.num_classes)
    # for w in range(len(weight)):
    #     if w in dataset.head_class_idx:
    #         weight[w] = 0
    # logger.info("Weight 0 applied to head classes", gpu_rank)
    trainer_ = Trainer(lr_scheduler=lr_scheduler, logger=logger, config=config, start_time=start_time,
                       gpu_rank=gpu_rank, device=device,
                       learning_model=learning_model,
                       train_dataset=train_dataset, test_dateset=test_dataset,
                       optimizer=optimizer)

    trainer_param = cfg['trainer']

    if 'checkpoint' in trainer_param:
        try:
            trainer_.load_checkpoint(trainer_param['checkpoint'])
            if 'ldam' in trainer_param['name']:
                classifier = get_attribute(learning_model, 'classifier')
                get_attribute(classifier, 'criterion', ddp_on=False).setup_start_epoch(trainer_.start_epoch)
            if 'ride' in trainer_param['name']:
                get_attribute(learning_model, 'criterion', ddp_on=dist.is_initialized()
                              ).setup_start_epoch(trainer_.start_epoch)
        except FileNotFoundError as e:
            logger.info(e)
            exit(-1)


    elif trainer_param['mode'] in ['train']:
        saved_values = trainer_.train_model()
        if config.output(gpu_rank) and len(trainer_param['record_more_details']) == 0:
            plot_results(saved_values, config)
    else:
        # plot_num_sample_per_cls(train_dataset)
        # visualise_prob_rate(dataset=train_dataset, method='bs')
        # visualise_class_prob(data_types=['train_b'], dataset=train_dataset,
        #                      methods=['res'])
        check_probs_target_class(train_dataset)
        # for method in ['res']:
        #     check_conflict(train_dataset, method=method, logger=logger)
        #     quantify_model_preference(train_dataset, method=method)
