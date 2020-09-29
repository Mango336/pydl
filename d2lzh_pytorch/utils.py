'''
封装的是线性回归中所需要的函数
linearTest.ipynb中import调用
'''

import torch
from IPython import display
from matplotlib import pyplot as plt
import numpy as np
import random

def use_svg_display():
    # 用矢量图显示
    display.set_matplotlib_formats('svg')

def set_figsize(figsize=(3.5, 2.5)):
    use_svg_display()
    # 设置图的尺寸
    plt.rcParams['figure.figsize'] = figsize

# 随机选取batch_size个特征和标签
def data_iter(batch_size, features, labels):
    num_examples = len(features)  # 特征变量的个数
    indices = list(range(num_examples))  # list中为 0--num_examples
    random.shuffle(indices)  # 样本的读取顺序是随机的 =》 将list（即indices）随机排序
    for i in range(0, num_examples, batch_size):
        j = torch.LongTensor(indices[i: min(i + batch_size, num_examples)])  # 最后一次可能不足一个batch
        yield features.index_select(0, j), labels.index_select(0, j)  # yield生成器 与return相似 但是仍有不同

# 线性回归矢量计算表达式
def linreg(X, w, b):
    return torch.mm(X, w) + b  # mm函数做矩阵乘法 样本点与权值相乘并+b

# 定义的损失函数 平方损失函数
def squared_loss(y_hat, y):
    return (y_hat - y.view(y_hat.size())) ** 2 / 2

# 优化算法 (这里是小批量随机梯度下降算法)
def sgd(params, lr, batch_size):
    for param in params:
        param.data -= lr * param.grad / batch_size  # 批量 所以需要求均值
