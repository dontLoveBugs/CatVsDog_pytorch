# -*- coding: utf-8 -*-
"""
 @Time    : 2018/12/24 20:08
 @Author  : Wang Xin
 @Email   : wangxin_buaa@163.com
"""


import os

path = '/home/data/UnsupervisedDepth/wangixn/catvsdog/train'

d = os.listdir(path)

classes = []
for e in d:
    print(e.split('.')[0], e.split('.')[1])

    name = e.split('.')[0]

    if name not in classes:
        classes.append(name)

classes.sort()
print(classes)



