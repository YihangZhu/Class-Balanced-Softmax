import math


class LR_scheduler:
    def __init__(self, name, num_epochs, lr, warmup=None, steps=None):
        self.lr = lr
        self.num_epochs = num_epochs
        self.do_warmup = warmup
        if name == 'linear':
            self.update_func = self.adjust_learning_rate_linear
        elif name == 'cosine':
            self.update_func = self.adjust_learning_rate_cosine
        elif name == 'multi_step':
            if steps is None:
                self.steps = [160, 180]
            else:
                self.steps = list(steps)
            self.update_func = self.adjust_lr_schedule_multistep
        else:
            raise Exception(f"LR scheduler type incorrect: {name}")

    def update_lr(self, epoch, optimiser):
        new_lr, return_lr = self.warmup(epoch)
        if not return_lr:
            new_lr = self.update_func(epoch)
        if optimiser is not None:
            for param_group in optimiser.param_groups:
                param_group['lr'] = new_lr
        return new_lr

    def adjust_lr_schedule_multistep(self, epoch):
        if epoch >= self.steps[1]:
            lr = self.lr * 0.01
        elif epoch >= self.steps[0]:
            lr = self.lr * 0.1
        else:
            lr = self.lr
        return lr

    def adjust_learning_rate_linear(self, epoch):
        """Sets the learning rate"""
        if epoch >= 0.75 * self.num_epochs:
            lr = self.lr * 0.01
        elif epoch >= 0.5 * self.num_epochs:
            lr = self.lr * 0.1
        else:
            lr = self.lr
        return lr

    def warmup(self, epoch):  # epoch is from 0 to num_epoch-1.
        if self.do_warmup:
            if epoch < self.do_warmup:
                return self.lr * ((epoch + 1) / self.do_warmup), True
        return self.lr, False

    def adjust_learning_rate_cosine(self, epoch):
        """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
        lr_min = 0
        lr_max = self.lr
        # 1+ is for when epoch equals 0 and lr will be lr_max
        lr = lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(epoch / self.num_epochs * math.pi))
        return lr


def linear(*args, **kwargs):
    return LR_scheduler(*args, name='linear', **kwargs)


def cosine(*args, **kwargs):
    return LR_scheduler(*args, name='cosine', **kwargs)


def multi_step(*args, **kwargs):
    return LR_scheduler(*args, name='multi_step', **kwargs)
