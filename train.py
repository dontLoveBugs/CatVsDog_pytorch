# -*- coding: utf-8 -*-
"""
 @Time    : 2018/12/24 20:08
 @Author  : Wang Xin
 @Email   : wangxin_buaa@163.com
"""
import csv
import os
import shutil
import socket
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.optim import lr_scheduler

import network

from torchvision.transforms import Resize, transforms

import utils
from dataloader import dataset
from utils import calculate_accuracy, AverageMeter

label = ['Cat', 'Dog']


def parse_command():
    import argparse
    parser = argparse.ArgumentParser(description='CatVsDog')
    parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet101',
                        help='model architecture: (default: resnet18)')
    parser.add_argument('--data_path', default='/home/data/UnsupervisedDepth/wangixn/catvsdog', help='dataset path')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 10)')
    parser.add_argument('--epochs', default=100, type=int, metavar='N',
                        help='number of total epochs to run (default: 15)')
    parser.add_argument('-b', '--batch-size', default=128, type=int, help='mini-batch size (default: 8)')
    parser.add_argument('--lr', '--learning-rate', default=0.0001, type=float,
                        metavar='LR', help='initial learning rate (default 0.01)')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=1e-5, type=float,
                        metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument(
        '--lr_patience',
        default=10,
        type=int,
        help='Patience of LR scheduler. See documentation of ReduceLROnPlateau.'
    )
    parser.add_argument(
        '--checkpoint',
        default=1,
        type=int,
        help='Trained model is saved at every this epochs.')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: checkpoint-5.pth.tar, model_best.pth.tar)')
    parser.add_argument('-e', '--evaluate', dest='evaluate', type=str, default='',
                        help='evaluate model on validation set')
    parser.add_argument('--no-pretrain', dest='pretrained', action='store_false',
                        help='not to use ImageNet pre-trained weights')
    parser.set_defaults(pretrained=True)
    args = parser.parse_args()
    return args


def create_dataloader(args):
    train_transform = transforms.Compose([
        Resize(128),
        transforms.RandomCrop((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    test_transform = transforms.Compose([
        Resize(128),
        transforms.CenterCrop((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    train_path = os.path.join(args.data_path, 'train')
    test_path = os.path.join(args.data_path, 'test')
    train_dataset = dataset.ImageFolder(root=train_path, ground_truth=True, transform=train_transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    test_dataset = dataset.ImageFolder(root=test_path, ground_truth=False, transform=test_transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)

    return train_loader, test_loader


def main():
    args = parse_command()

    if torch.cuda.device_count() > 1:
        args.batch_size = args.batch_size * 2
        print('Using GPUs:', 2)

    model = network.CatDogClassifier()
    train_params = [{'params': network.get_1x_lr_params(model), 'lr': args.lr},
                    {'params': network.get_10x_lr_params(model), 'lr': args.lr * 10}]

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model).cuda()
    elif torch.cuda.is_available():
        model = model.cuda()

    criterion = nn.CrossEntropyLoss()
    if torch.cuda.is_available():
        criterion = criterion.cuda()

    optimizer = optim.SGD(train_params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min', patience=args.lr_patience)

    # loader
    train_loader, test_loader = create_dataloader(args)

    # create results folder, if not already exists
    output_directory = utils.get_output_directory_run(args)
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    log_path = os.path.join(output_directory, 'logs',
                            datetime.now().strftime('%b%d_%H-%M-%S') + '_' + socket.gethostname())
    if os.path.isdir(log_path):
        shutil.rmtree(log_path)
    os.makedirs(log_path)
    logger = SummaryWriter(log_path)

    for epoch in range(args.epochs):
        acc = train(epoch, model, train_loader, optimizer, criterion, logger)
        test(epoch, model, test_loader, output_directory)
        scheduler.step(acc)

        for i, param_group in enumerate(optimizer.param_groups):
            old_lr = float(param_group['lr'])
            # print(i, old_lr)

            logger.add_scalar('Lr/lr_' + str(i), old_lr, epoch)

        if i % args.checkpoint == 0:
            save_file_path = os.path.join(output_directory,
                                          'save_{}.pth'.format(i))
            states = {
                'epoch': i + 1,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }
            torch.save(states, save_file_path)


def train(epoch, model, data_loader, optimizer, criterion, logger):
    model.train()

    losses = AverageMeter()
    accuracies = AverageMeter()

    for i, data in enumerate(data_loader):
        inputs, labels = data

        if torch.cuda.is_available():
            inputs = inputs.cuda()
            labels = labels.cuda()

        torch.cuda.synchronize()

        optimizer.zero_grad()
        outputs = model(inputs)

        loss = criterion(outputs, labels)
        acc = calculate_accuracy(outputs, labels)
        losses.update(loss.data[0], inputs.size(0))
        accuracies.update(acc, inputs.size(0))

        loss.backward()
        optimizer.step()

        torch.cuda.synchronize()

        print('Train: Epoch: [{0}][{1}/{2}]\t'
              'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
              'Acc {acc.val:.3f} ({acc.avg:.3f})'.format(
            epoch,
            i + 1,
            len(data_loader),
            loss=losses,
            acc=accuracies))

        if logger is not None:
            current_step = epoch * len(data_loader) + i
            logger.add_scalar('Train/Loss', losses.avg, current_step)
            logger.add_scalar('Train/Acc', accuracies.avg, current_step)

    if logger is not None:
        logger.add_scalar('Train/Loss_epoch', losses.avg, epoch)
        logger.add_scalar('Train/Acc_epoch', accuracies.avg, epoch)

    return accuracies.avg


def test(epoch, model, data_loader,  output_directory, write_to_file=True):
    print('Test at epoch {}'.format(epoch))

    fieldnames = ['id', 'label']

    if write_to_file:
        filename = 'test-' + str(epoch) + '.csv'
        test_csv = os.path.join(output_directory, filename)
        with open(test_csv, 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

    # losses = AverageMeter()
    # accuracies = AverageMeter()

    model.eval()

    for i, data in enumerate(data_loader):
        inputs, idx = data

        if torch.cuda.is_available():
            inputs = inputs.cuda()

        torch.cuda.synchronize()

        with torch.no_grad():
            outputs = model(inputs)

        # loss = criterion(outputs, labels)
        # acc = calculate_accuracy(outputs, labels)
        # losses.update(loss.data[0], inputs.size(0))
        # accuracies.update(acc, inputs.size(0))

        torch.cuda.synchronize()

        # print('Test: Epoch: [{0}][{1}/{2}]\t'
        #       'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
        #       'Acc {acc.val:.3f} ({acc.avg:.3f})'.format(
        #     epoch,
        #     i + 1,
        #     len(data_loader),
        #     loss=losses,
        #     acc=accuracies))

        _, pred = outputs.topk(1, 1, True)
        pred = pred.t()
        # print('pred: ', pred.item())

        if write_to_file:
            with open(test_csv, 'a') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writerow({'id': idx[0], 'label': label[pred.item()]})

    print('Test finished at epoch {}'.format(epoch))
    # if logger is not None:
    #     logger.add_scalar('Test/loss', losses.avg, epoch)
    #     logger.add_scalar('Test/Acc', accuracies.avg, epoch)


if __name__ == '__main__':
    main()
