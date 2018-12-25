# -*- coding: utf-8 -*-
"""
 @Time    : 2018/12/25 12:49
 @Author  : Wang Xin
 @Email   : wangxin_buaa@163.com
"""
import csv
import os

import torch
import torch.nn as nn

from torchvision.transforms import Resize, transforms

import utils
import network
from dataloader import dataset

label = ['Cat', 'Dog']


def parse_command():
    import argparse
    parser = argparse.ArgumentParser(description='CatVsDog')
    parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet101',
                        help='model architecture: (default: resnet18)')
    parser.add_argument('--data_path', default='/home/data/UnsupervisedDepth/wangixn/catvsdog', help='dataset path')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 10)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: checkpoint-5.pth.tar, model_best.pth.tar)')
    args = parser.parse_args()
    return args


def create_dataloader(args):
    test_transform = transforms.Compose([
        Resize(128),
        transforms.CenterCrop((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    test_path = os.path.join(args.data_path, 'test')

    test_dataset = dataset.ImageFolder(root=test_path, transform=test_transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)

    return test_loader


def main():
    args = parse_command()
    test_loader = create_dataloader(args)

    model = network.CatDogClassifier()

    model = nn.DataParallel(model)
    if torch.cuda.is_available():
        model = model.cuda()

    print('loading pretrained model {}'.format(args.resume))
    pretrain = torch.load(args.resume)
    model.load_state_dict(pretrain['model'])
    epoch = pretrain['epoch']
    del pretrain  # 清理缓存

    criterion = nn.CrossEntropyLoss()
    if torch.cuda.is_available():
        criterion = criterion.cuda()

    # create results folder, if not already exists
    output_directory = utils.get_output_directory_run(args)
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    test(epoch, model, test_loader, output_directory)


def test(epoch, model, data_loader, output_directory, write_to_file=True):
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
        inputs, labels = data

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
        print('pred: ', pred.item())

        if write_to_file:
            with open(test_csv, 'a') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writerow({'id': i, 'label': label[pred.item()]})


if __name__ == '__main__':
    main()
