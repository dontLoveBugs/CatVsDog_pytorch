# -*- coding: utf-8 -*-
"""
 @Time    : 2018/12/24 21:16
 @Author  : Wang Xin
 @Email   : wangxin_buaa@163.com
"""
import glob
import os


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def get_fine_tuning_parameters(model, ft_begin_index):
    if ft_begin_index == 0:
        return model.parameters()

    ft_module_names = []
    for i in range(ft_begin_index, 5):
        ft_module_names.append('layer{}'.format(i))
    ft_module_names.append('fc')

    parameters = []
    for k, v in model.named_parameters():
        for ft_module in ft_module_names:
            if ft_module in k:
                parameters.append({'params': v})
                break
        else:
            parameters.append({'params': v, 'lr': 0.0})

    return parameters


def calculate_accuracy(outputs, targets):
    batch_size = targets.size(0)

    _, pred = outputs.topk(1, 1, True)
    pred = pred.t()
    correct = pred.eq(targets.view(1, -1))
    n_correct_elems = correct.float().sum().data[0]

    return n_correct_elems / batch_size


def get_output_directory_run(args):
    save_dir_root = os.path.join(os.path.dirname(os.path.abspath(__file__)))
    if args.resume:
        return args.resume
    else:
        runs = sorted(glob.glob(os.path.join(save_dir_root, 'run', 'run_*')))
        run_id = int(runs[-1].split('_')[-1]) + 1 if runs else 0
    save_dir = os.path.join(save_dir_root, 'run', 'run_' + str(run_id))

    return save_dir
