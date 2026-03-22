import os
import pickle
import shutil
import time
from collections import OrderedDict

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.cuda.amp import GradScaler
from torch.cuda.amp import autocast

from utils.compute import adjust_loss, norm_linear_weight_grad, ce_grad_func
from utils.meter import AverageMeter, ImbalanceAccuracy, ProgressMeter, TrainRecorder, check_linear_classifier
from utils.utils import get_attributes, check_file, reduce_all, get_attribute, check_identical


class Trainer:
    def __init__(self, lr_scheduler, logger, config, start_time, gpu_rank, device,
                 learning_model=None, optimizer=None, train_dataset=None, test_dateset=None):
        self.start_time = start_time
        self.learning_task = config.config['trainer']['learning_task']
        self.start_epoch = config.config['trainer']['start_epoch']
        self.pre_runtime = 0

        self.gpu_rank = gpu_rank
        self.device = device

        self.ddp_on = dist.is_initialized()
        self.model_update = config.config['trainer']["model_update"]
        self.run_test = config.config['trainer']['run_test']
        if test_dateset is not None:
            self.test_dataset = test_dateset
        if learning_model is not None:
            self.learning_model = learning_model
            self.optimizer = optimizer
            self.train_dataset = train_dataset
            self.train_recorder = TrainRecorder(None if test_dateset is None else test_dateset.test_dataloader.keys())

            if 'ce_grads' in config.config['trainer']['record_more_details'] or 'scl_grads' in config.config['trainer'][
                'record_more_details']:
                func = get_attribute(learning_model, 'set_check_grad', self.ddp_on)
                if func is not None:
                    func()

        self.lr_scheduler = lr_scheduler
        self.logger = logger
        self.config = config
        self.run_epochs = self.config.config['trainer']['run_epochs']
        if "save_all_ckp" in self.config.config['trainer']:
            self.run_epochs = self.config.config['trainer']['save_all_ckp']['epochs']

        self.is_last_epoch = self.start_epoch == self.run_epochs - 1
        self.final_eval = self.config.config['trainer']['final_test']

        self.scaler = GradScaler() if self.config.config['trainer']['fp16'] and torch.cuda.is_available() else None

    def get_runtime(self):
        return time.time() - self.start_time + self.pre_runtime

    def is_output(self):
        return self.config.output(self.gpu_rank)

    def load_checkpoint(self, file_path, model_only=False):
        if not self.config.config['trainer']['resume']:
            model_only = True

        def load_state(state):
            state = get_state_dict(state, self.ddp_on)
            try:
                self.learning_model.load_state_dict(
                    state, strict=True)
            except RuntimeError as e:
                self.logger.info(e, gpu_rank=self.gpu_rank)
                if f'mismatch' in e.__str__() and 'fc.weight' in e.__str__():
                    if 'module.' in e.__str__():
                        del state['module.classifier.fc.weight']
                        if 'module.classifier.fc.bias' in state:
                            del state['module.classifier.fc.bias']
                    else:
                        del state['classifier.fc.weight']
                        if 'classifier.fc.bias' in state:
                            del state['classifier.fc.bias']
                    self.logger.info('fc.weight, fc.bias are deleted from the state dict.', gpu_rank=self.gpu_rank)

                # if 'classifier.fc.weight' in state:
                #     del state['classifier.fc.weight']
                # elif 'module.classifier.fc.weight' in state:
                #     del state['module.classifier.fc.weight']
                self.learning_model.load_state_dict(
                    state, strict=False)

        if os.path.isfile(file_path):
            self.logger.info(f"=> loading checkpoint '{file_path}'", gpu_rank=self.gpu_rank)
            try:
                checkpoint = torch.load(file_path, map_location=self.device)
            except RuntimeError as e:
                self.logger.info(e)
                with open(file_path, 'rb') as f:
                    checkpoint = pickle.load(f)
            # set strict to True for omitting key missmatch in the checkpoint and the current model
            if 'state_dict_model' not in checkpoint:
                load_state(checkpoint)
                self.logger.info(f"=> loaded checkpoint '{file_path}'", gpu_rank=self.gpu_rank)
            else:
                load_state(checkpoint['state_dict_model'])
                if model_only:
                    self.logger.info(f"=> loaded checkpoint '{file_path}'", gpu_rank=self.gpu_rank)
                    del checkpoint
                    return
                self.start_epoch = checkpoint['epoch'] + 1
                if 'saved_values' in checkpoint:
                    self.train_recorder.update_all_record(checkpoint['saved_values'], self.logger, self.gpu_rank)
                self.pre_runtime = checkpoint['runtime']
                self.logger.info(f"=> loaded checkpoint '{file_path}' (trained {checkpoint['epoch']}+1 epochs)",
                                 gpu_rank=self.gpu_rank)
        else:
            info = f"=> no checkpoint found at '{file_path}'"
            self.logger.info(info, gpu_rank=self.gpu_rank)
            raise FileNotFoundError(info)

    def check_model_evolution(self, checkpoint_root):
        def process_ckps():
            self.load_checkpoint(path, model_only=True)
            self.logger.info(f"=> loaded checkpoint '{path}'", gpu_rank=self.gpu_rank)
            linear_classifier_data = check_linear_classifier(self.learning_model, self.train_dataset,
                                                             self.logger, self.ddp_on, self.gpu_rank,
                                                             rate=self.config.config['trainer']['classifier_rate'])
            self.train_recorder.update_linear_classifier_record(linear_classifier_data)

        for epoch in range(self.run_epochs):
            if epoch < self.config.config['trainer']['save_all_ckp']['epochs']:
                i = 0
                while True:
                    path = f'{checkpoint_root}/epoch_{epoch}_{i}_ckp.tar'
                    if check_file(path):
                        process_ckps()
                        i += self.config.config['trainer']['save_all_ckp']['step']
                    else:
                        break
            else:
                break
            path = f'{checkpoint_root}/epoch_{epoch}__ckp.tar'
            process_ckps()

        self.logger.info(f'All checkpoints processed', gpu_rank=self.gpu_rank)
        return self.get_recorded_values()

    def train_model(self):
        self.logger.info(f'Preparation time: {self.get_runtime():.3f}', gpu_rank=self.gpu_rank)
        self.logger.info(f'>>===========Training started===========>>', gpu_rank=self.gpu_rank)
        epoch_time = time.time()

        for epoch in range(self.start_epoch, self.run_epochs):
            self.is_last_epoch = epoch == self.run_epochs - 1
            # training results will be different for models on different GPUs,
            # because different GPUs use different batches of data.
            if hasattr(self.train_dataset, 'update_train_dataset'):
                self.train_dataset.update_train_dataset()
            self.train(epoch)

            if hasattr(self, 'test_dataset') and self.run_test and (not self.is_last_epoch):
                self.validate(epoch=epoch)
            self.logger.info(f'Epoch_time: {time.time() - epoch_time:.3f}', gpu_rank=self.gpu_rank)

            # linear_classifier_data = check_linear_classifier(self.learning_model, self.train_dataset,
            #                                                  self.logger, self.ddp_on, self.gpu_rank)
            # if linear_classifier_data is not None:
            #     self.train_recorder.update_linear_classifier_record(linear_classifier_data)

            # when running with multiple GPUs, all the models in different GPUs will be identical,
            # therefore, evaluation results are identical.

            # print(self.saved_values)
            if self.is_output() and self.model_update:
                self.save_checkpoint(epoch, self.get_runtime(), self.is_last_epoch,
                                     all_epoch="save_all_ckp" in self.config.config['trainer'])
            epoch_time = time.time()
        if self.final_eval and hasattr(self, 'test_dataset'):
            self.is_last_epoch = True
            self.validate(epoch=self.run_epochs - 1)

        return self.get_recorded_values()

    def get_recorded_values(self):
        return {"": self.train_recorder.get_saved_values(clean=True)}

    def train_per_epoch(self, epoch, model, dataset, optimiser, scheduler, recorder, prefix='Train'):
        if self.ddp_on:
            dataset.distributed_sampler.set_epoch(epoch)  # the sampler is a distributed sampler

        lr = scheduler.update_lr(epoch, optimiser) if scheduler else None
        # switch to train mode
        model.train()
        t_report, metrics_values, loss = self.process_epoch(model,
                                                            dataset.train_dataloader,
                                                            dataset,
                                                            optimizer=optimiser, epoch=epoch,
                                                            prefix=prefix)
        recorder.update_train_record(t_report, metrics_values, loss)

        """training accuracy may be slightly different between different GPUs"""
        self.logger.info(f"Runtime: {self.get_runtime():.3f}\tlr: {lr}", gpu_rank=self.gpu_rank)

    def train(self, epoch):
        self.train_per_epoch(epoch, self.learning_model, self.train_dataset, self.optimizer, self.lr_scheduler,
                             self.train_recorder)

    def validate_per_epoch(self, epoch, model, dataset, recorder, prefix='Eval'):
        model.eval()
        for test_type, test_dataloader in dataset.test_dataloader.items():
            with torch.no_grad():
                v_report, metrics_values, loss = self.process_epoch(model,
                                                                    test_dataloader,
                                                                    dataset,
                                                                    optimizer=None,
                                                                    epoch=epoch, prefix=prefix)

                recorder.update_eval_record(v_report, metrics_values, test_type, loss)
                self.logger.info(f"Runtime: {self.get_runtime():.3f}\t", gpu_rank=self.gpu_rank)

    def validate(self, epoch=0):
        self.validate_per_epoch(epoch, self.learning_model, self.test_dataset, self.train_recorder)
        if self.is_last_epoch:
            if hasattr(self.test_dataset, 'evaluate'):
                self.test_dataset.evaluate(logger=self.logger, gpu_rank=self.gpu_rank)

    def add_additional_loss_components(self, all_loss, learning_model):
        funcs = get_attributes(learning_model, 'get_loss_components', self.ddp_on)
        for func in funcs:
            all_loss.extend(func())
        return all_loss

    def pre_epoch_setup(self, epoch, learning_model):
        funcs = get_attributes(learning_model, 'pre_epoch_setup', self.ddp_on)
        for func in funcs:
            func(epoch, self.device)

    def post_epoch_setup(self, learning_model):
        funcs = get_attributes(learning_model, 'end_epoch_process', self.ddp_on)
        for func in funcs:
            func(self.logger, self.gpu_rank)

    def process_epoch(self, learning_model, dataloader, dataset, optimizer, epoch, prefix):
        prefix = f"{prefix} epoch[{epoch}]: "
        change_prefix = True
        batch_time = AverageMeter('Batch_time', ':6.3f')
        data_time = AverageMeter('Data_time', ':6.3f')
        loss = AverageMeter('Loss', ':.3f')
        all_loss = self.add_additional_loss_components([loss], learning_model)
        meters = [batch_time, data_time, *all_loss]

        progress = ProgressMeter(
            len(dataloader),
            meters,
            prefix=f"{prefix}",
            print_freq=self.config.config['trainer']['print_freq'])
        imb_process = ImbalanceAccuracy(dataset, self.device) if self.learning_task == 'classification' else None
        is_train = 'Train' in prefix
        num_samples = 0
        end = time.time()
        # print(f"Start running..{device}")
        # sample_num_count = {}
        # for i in range(self.train_dataset.num_classes):
        #     sample_num_count[i] = 0
        self.pre_epoch_setup(epoch, learning_model)

        for i, (input_data) in enumerate(dataloader):
            samples, targets = input_data[0], input_data[1]
            if isinstance(samples, list):
                samples = torch.cat([sample for sample in samples], dim=0)
                targets = torch.cat([target for target in targets], dim=0)
            # array_check = images.detach().numpy()

            # count_targets = targets.detach().numpy()
            # num_cls = Counter(count_targets)
            # for key, value in num_cls.items():
            #     sample_num_count[key] += value

            # measure data loading time
            data_time.update(time.time() - end)
            if isinstance(samples, torch.Tensor):  # when using SAM the samples are dicts instead of tensor
                samples = samples.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)
            if self.scaler is not None:
                with autocast():
                    output = learning_model(samples, targets)
            else:
                output = learning_model(samples, targets)
            # measure accuracy and record loss

            # (acc1_over_sample, acc5_over_sample), pred = accuracy(
            # output['output'], targets, topk=(1, min(self.dataset.num_classes, 5)))
            # imb_process.update(targets, predicted=pred[0, :])
            if 'targets' not in output:
                output['targets'] = targets
            if imb_process is not None:
                if "similarity_b" in output:
                    imb_process.update(target=output['targets'], output=output['similarity_b'])
                    if change_prefix:
                        prefix = prefix.replace('Train', 'Train_B')
                        change_prefix = False
                else:
                    imb_process.update(target=output['targets'], output=output['similarities'])

            if self.is_last_epoch and not is_train:
                if hasattr(self.test_dataset, 'record_results'):
                    self.test_dataset.record_results(output['similarities'], input_data[2])

            if 'loss' in output:
                loss.update(output['loss'].item())
            if 'target_prob' in self.config.config['trainer']["record_more_details"]:
                assert not self.model_update
                saved_values = self.train_recorder.get_saved_values()
                probs = torch.softmax(output['similarities'].detach(), dim=1)
                if is_train:
                    if 'target_prob' not in saved_values:
                        saved_values['target_prob'] = {}
                        saved_values['target_prob']['targets'] = []
                        saved_values['target_prob']['probs'] = []
                    saved_values['target_prob']['targets'].extend(output['targets'].detach())
                    target_prob = probs[torch.arange(probs.size(0)), output['targets'].detach()]
                    saved_values['target_prob']['probs'].extend(target_prob)


            # used for calculating class probs
            if 'class_prob' in self.config.config['trainer']["record_more_details"]:
                assert not self.model_update
                probs = torch.softmax(output['similarities'].detach(), dim=1)
                num_samples += probs.size(0)
                saved_values = self.train_recorder.get_saved_values()
                if is_train:
                    if 'class_prob_train' not in saved_values:
                        saved_values['class_prob_train'] = torch.zeros_like(probs[0])
                    saved_values['class_prob_train'] += torch.sum(probs, dim=0)
                    if 'similarity_b' in output:
                        probs_b = torch.softmax(output['similarity_b'], dim=1)
                        if 'class_prob_b_train' not in saved_values:
                            saved_values['class_prob_b_train'] = torch.zeros_like(probs[0])
                        saved_values['class_prob_b_train'] += torch.sum(probs_b, dim=0)
                else:
                    assert 'Eval' in prefix
                    if 'class_prob_eval' not in saved_values:
                        saved_values['class_prob_eval'] = torch.zeros_like(probs[0])
                    saved_values['class_prob_eval'] += torch.sum(probs, dim=0)

            if optimizer is not None:
                optimizer.zero_grad()
                learning_model.zero_grad()
                retain_graph = False

                if 'retain_graph' in output:
                    retain_graph = output['retain_graph']

                if 'loss' in output:
                    if self.config.config['trainer']['adjust_loss']:
                        output['loss'] = adjust_loss(output['loss'], self.logger)

                    z = output['loss']
                    z_grad = None
                else:
                    z = output['similarities']
                    z_grad = output['logit_grad']

                if self.scaler is None:
                    z.backward(z_grad, retain_graph=retain_graph)
                else:
                    self.scaler.scale(z).backward(z_grad, retain_graph=retain_graph)

                if 'ce_grads' in self.config.config['trainer']["record_more_details"] and is_train:
                    # self.record_more_details(dataset, output)
                    self.calculate_cross_entropy_loss_grads(output, dataset, learning_model,
                                                            beta=output['grad_beta'] if 'grad_beta' in output else 1,
                                                            check=not torch.cuda.is_available()
                                                            )
                if self.model_update:
                    if self.scaler is None:
                        optimizer.step()
                    else:
                        self.scaler.step(optimizer)
                        self.scaler.update()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            progress.display(i, self.logger, self.gpu_rank)

            del output

        all_loss_recorded = {}
        for item in all_loss:
            key = item.name.replace('_Loss', '').replace('Loss', '')
            all_loss_recorded[key] = item.avg
        if self.is_last_epoch:
            saved_values = self.train_recorder.get_saved_values()

            def store_target_prob():
                if 'target_prob' in saved_values:
                    if self.ddp_on and is_train:
                        from utils.utils import concat_all_gather
                        saved_values['target_prob']['probs'] = concat_all_gather(saved_values['target_prob']['probs'])
                        saved_values['target_prob']['targets'] = concat_all_gather(saved_values['target_prob']['targets'])
                    torch.save(saved_values['target_prob'], self.config.dirs['save_path'] / f'target_probs_{self.gpu_rank}.pk')

            def store_class_prob(title):
                if title in saved_values:
                    if self.ddp_on and is_train:
                        saved_values[title] = reduce_all(saved_values[title], 'SUM', self.logger)
                    saved_values[title] = saved_values[title] / num_samples
                    torch.save(saved_values[title], self.config.dirs['save_path'] / f'{title}.pk')
                    del saved_values[title]

            def save_grads(key_name, reduction='SUM'):
                if key_name in saved_values:
                    if self.ddp_on:
                        for key, value in saved_values[key_name].items():
                            saved_values[key_name][key] = reduce_all(value, reduction, self.logger)
                    grads = {
                        key_name: saved_values[key_name]
                    }
                    torch.save(grads, self.config.dirs['save_path'] / f'{key_name}.pk')
                    del saved_values[key_name]


            store_target_prob()
            store_class_prob('class_prob_train')
            store_class_prob('class_prob_eval')
            store_class_prob('class_prob_b_train')
            save_grads('ce_grads')
        self.post_epoch_setup(learning_model)

        if imb_process is not None:
            report, metrics_values, detailed_values = imb_process.calculate(last_epoch=self.is_last_epoch,
                                                                            use_ddp=self.ddp_on and is_train,
                                                                            logger=self.logger)
            self.config.logger.info(f"{prefix}{report}", gpu_rank=self.gpu_rank)
            if self.is_last_epoch and self.is_output():
                prefix_values = 'train' if is_train else 'test'
                torch.save(detailed_values, f"{self.config.dirs['save_path']}/{prefix_values}_class_values.pk")

            return report, metrics_values, all_loss_recorded
        else:
            return None, None, all_loss_recorded

    def calculate_cross_entropy_loss_grads(self, output, dataset, learning_model, check=True, beta=1, norm_ln=True):
        logit_grad, (logit_reward_grad, logit_penalty_grad) = ce_grad_func(
            output['similarities'], output['targets'], dataset.num_classes, check=check
        )

        b_A = logit_reward_grad.sum(dim=0)
        b_B = logit_penalty_grad.sum(dim=0)

        w_reward_grads = torch.mm(logit_reward_grad.T, output['feature'])
        w_penalty_grades = torch.mm(logit_penalty_grad.T, output['feature'])

        w_A = w_reward_grads
        w_B = w_penalty_grades

        grad_w = None

        classifier = get_attribute(learning_model, 'classifier', self.ddp_on)
        if classifier is None:
            fc = get_attribute(learning_model, 'fc', self.ddp_on)
        else:
            fc = classifier.fc
        if norm_ln:
            manual_w_grad = norm_linear_weight_grad(output['feature'], logit_grad, fc.weight)

            if "scl_loss" in output and 'detach' not in self.config.config['learning_model']['args']['class_complement']:
                grad_w = torch.autograd.grad(
                    outputs=output["scl_loss"],
                    inputs=fc.weight,
                    retain_graph=True
                )[0]
                manual_w_grad += grad_w
        else:
            manual_w_grad = (w_A + w_B)

        if check:
            check_identical(manual_w_grad, fc.weight.grad, "Weight ")

        bias_grads = (b_A + b_B)
        if hasattr(fc, 'bias') and fc.bias is not None:
            check_identical(bias_grads, fc.bias.grad, "Bias ")

        saved_values = self.train_recorder.get_saved_values()
        if 'ce_grads' not in saved_values:
            saved_values['ce_grads'] = dict()
            saved_values['ce_grads']['bias_grads_reward'] = torch.zeros_like(b_A)
            saved_values['ce_grads']['bias_grads_penalty'] = torch.zeros_like(b_A)
            saved_values['ce_grads']['weight_grads_reward'] = torch.zeros_like(w_A)
            saved_values['ce_grads']['weight_grads_penalty'] = torch.zeros_like(w_A)
            if grad_w is not None:
                saved_values['ce_grads']['grad_scl_weight'] = torch.zeros_like(grad_w)

        grads = saved_values['ce_grads']

        grads['bias_grads_reward'] += b_A.detach()
        grads['bias_grads_penalty'] += b_B.detach()
        grads['weight_grads_reward'] += w_A.detach()
        grads['weight_grads_penalty'] += w_B.detach()
        if grad_w is not None:
            grads['grad_scl_weight'] += grad_w.detach()

    def record_more_details(self, dataset, output):
        targets = output['targets']
        target_one_hot = torch.nn.functional.one_hot(targets, dataset.num_classes)
        probs = torch.softmax(output['similarities'], dim=1)
        logits = torch.sum(output['similarities'] * target_one_hot, dim=1)
        sample_classes = set(list(targets.detach().cpu().numpy()))
        saved_values = self.train_recorder.get_saved_values()
        for target in sample_classes:
            mask = targets == target
            prob_selected = probs[mask]
            z = logits[mask]
            features_selected = output['feature'][mask]

            if 'probs' not in saved_values:
                saved_values['probs'] = [[] for _ in range(dataset.num_classes)]
            saved_values['probs'][target].extend(prob_selected.detach().cpu().numpy())

            if 'logits' not in saved_values:
                saved_values['logits'] = [[] for _ in range(dataset.num_classes)]
            saved_values['logits'][target].extend(z.detach().cpu().numpy())

            if 'features' not in saved_values:
                saved_values['features'] = [[] for _ in range(dataset.num_classes)]
            saved_values['features'][target].extend(features_selected.detach().cpu().numpy())

    def get_checkpoint(self, epoch, runtime):
        checkpoint = {
            'epoch': epoch,
            'state_dict_model': self.learning_model.state_dict(),
            'runtime': runtime,
            'saved_values': self.train_recorder.get_saved_values()
        }
        return checkpoint

    def save_checkpoint(self, epoch, runtime, is_last=None, all_epoch=False, batch_id=None):
        checkpoint = self.get_checkpoint(epoch, runtime)
        model_dir = self.config.dirs['model_dir']

        if all_epoch:
            name = f'epoch_{epoch}_{batch_id}_ckp.tar' if batch_id is not None else f'epoch_{epoch}_ckp.tar'
            torch.save(checkpoint, model_dir / name)
        else:
            torch.save(checkpoint, model_dir / 'current_ckp.tar')
            if is_last:
                shutil.move(model_dir / 'current_ckp.tar', model_dir / 'last_ckp.tar')


def get_state_dict(state_dict, ddp):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if ddp:
            if 'module.' in k:
                return state_dict
            else:
                name = 'module.' + k  # remove 'module.' of DataParallel/DistributedDataParallel
                new_state_dict[name] = v
        else:
            if 'module.' in k:
                name = k[7:]  # remove 'module.' of DataParallel/DistributedDataParallel
                new_state_dict[name] = v
            else:
                return state_dict
    state_dict = new_state_dict
    return state_dict
