# -*- coding: utf-8 -*-
"""
 @Time    : 2018/12/24 19:13
 @Author  : Wang Xin
 @Email   : wangxin_buaa@163.com
"""

import csv

res_files = '/home/data/UnsupervisedDepth/wangixn/Project/CatVsDog_pytorch/run/run_6/test-17.csv'
example_file = '/home/data/UnsupervisedDepth/wangixn/Project/CatVsDog_pytorch/run/run_0/submission_example.csv'
sub_file = '/home/data/UnsupervisedDepth/wangixn/Project/CatVsDog_pytorch/run/run_6/submission.csv'
# results = csv.reader(open(res_files, 'r'))
examples = csv.reader(open(example_file, 'r'))


fieldnames = ['id', 'label']

with open(sub_file, 'w') as sub_csv:
    writer = csv.DictWriter(sub_csv, fieldnames=fieldnames)
    writer.writeheader()

def find(x):
    results = csv.reader(open(res_files, 'r'))
    for e in results:
        # print(x, e[0])
        if e[0] == x:
            return e[1]

    return -1


for e in examples:
    if e[0] == 'id':
        continue

    x = e[0]
    y = find(x)

    if y == -1:
        print('no find:', x)
    else:
        print('find:', x)

    with open(sub_file, 'a') as sub_csv:
        writer = csv.DictWriter(sub_csv, fieldnames=fieldnames)
        writer.writerow({'id': x, 'label': y})
