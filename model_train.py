import copy
import time

import torch
from torchvision.datasets import FashionMNIST
from torchvision import transforms
import torch.utils.data as Data
import numpy as np
import matplotlib.pyplot as plt
from model import LeNet
import torch.nn as nn
import pandas as pd

# 完成数据集的读取、划分和封装，最后返回两个 DataLoader
def train_val_data_process():
    # 加载 FashionMNIST 数据集
    train_data = FashionMNIST(root='./data',
                              train=True,
                              transform=transforms.Compose([transforms.Resize(size=28), transforms.ToTensor()]),
                              download=True)
#     划分训练集和数据集
    train_data, val_data = torch.utils.data.random_split(train_data,
                                                         [round(0.8*len(train_data)),
                                                          round(0.2*len(train_data))])
#     转换成批次的数据集DataLoader 的作用是把数据集组织成可以按批次遍历的数据加载器
#     num_workers=2
    # 表示使用 2 个工作子进程加载数据。
    # 这些进程主要负责读取和预处理数据，以便主进程进行模型计算。
    train_dataloader = Data.DataLoader(dataset=train_data,
                                       batch_size=64,
                                       shuffle=True,
                                       num_workers=2)
    val_dataloader = Data.DataLoader(dataset=val_data,
                                     batch_size=64,
                                     shuffle=False,
                                     num_workers=2)
    return train_dataloader, val_dataloader

# model	需要训练的神经网络模型，例如 LeNet
# train_dataloader	用于读取训练集 batch 的数据加载器
# val_dataloader	用于读取验证集 batch 的数据加载器
# num_epochs	训练轮数
def model_train_process(model, train_dataloader, val_dataloader, num_epochs):
    # 定义训练要用的设备，有gpu用gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#   定义优化器，model.parameter是模型的参数，lr是学习率
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
#   定义损失函数，分类问题使用交叉熵损失函数
    criterion = nn.CrossEntropyLoss()
    #将模型放入训练设备
    model = model.to(device)
    #   复制当前模型的参数
    #     model.state_dict()
    #     获取模型的状态字典。
    #     它一般是一个类似字典的有序映射，其中保存模型的参数Tensor和持久化缓冲区
    #深拷贝。
    #
    # 创建一份独立的模型状态副本，避免后续训练更新参数时，已经保存的最佳参数也跟着变化。
    best_model_wts = copy.deepcopy(model.state_dict())


    # 初始化参数
    # 最高准确度
    best_acc = 0.0
    # 训练集损失列表
    train_loss_all = []
    # 验证集损失列表
    val_loss_all = []
    # 训练集准确度列表
    train_acc_all = []
    # 验证集准确度列表
    val_acc_all = []
    # 当前时间
    since = time.time()

    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch, num_epochs-1))
        print("-"*10)

        train_loss = 0.0
        train_corrects = 0

        val_loss = 0.0
        val_corrects = 0

        train_num = 0
        val_num = 0

        # 设置模型为训练模式
        model.train()
        # 对每一个mini-batch训练和计算
        for step, (b_x, b_y) in enumerate(train_dataloader):
            # 将特征放入到训练设备中
            b_x = b_x.to(device)
            # 将标签放入到训练设备中
            b_y = b_y.to(device)


            # 前向传播过程，输入为一个batch，输出为一个batch中对应的预测
            output = model(b_x)
            # 查找每一行中最大值对应的行标
            pre_lab = torch.argmax(output, dim=1)
            # 计算每一个batch的平均损失函数
            loss = criterion(output, b_y)

            # 将梯度初始化为0
            optimizer.zero_grad()
            # 反向传播计算
            loss.backward()
            # 根据网络反向传播的梯度信息来更新网络的参数，以起到降低loss函数计算值的作用
            optimizer.step()

            # 这一个批次里面的loss，对损失函数进行累加
            train_loss += loss.item() * b_x.size(0)
            # 如果预测正确，则准确度train_corrects加1
            #对布尔 Tensor 求和时，True 记为 1，False 记为 0
            train_corrects += torch.sum(pre_lab == b_y.data)
            # 当前用于训练的样本数量
            train_num += b_x.size(0)

        model.eval()
        for step, (b_x, b_y) in enumerate(val_dataloader):
            b_x = b_x.to(device)
            b_y = b_y.to(device)


            output = model(b_x)
            pre_lab = torch.argmax(output, dim=1)
            loss = criterion(output, b_y)
            # 验证集只是用于评估模型性能，不应该使用验证集的损失来直接更新模型参数。

            val_loss += loss.item() * b_x.size(0)
            val_corrects += torch.sum(pre_lab == b_y.data)
            val_num += b_x.size(0)

        # 计算并保存每一个epoch的loss值和准确率
        # 计算并保存训练集的loss值
        train_loss_all.append(train_loss / train_num)
        # 计算并保存训练集的准确率
        train_acc_all.append(train_corrects.double().item() / train_num)

        # 计算并保存验证集的loss值
        val_loss_all.append(val_loss / val_num)
        # 计算并保存验证集的准确率
        val_acc_all.append(val_corrects.double().item() / val_num)

        # 打印每一轮的loss和accuracy
        print("train Loss: {:.4f} Acc: {:.4f}".format(train_loss_all[-1], train_acc_all[-1]))
        print("val Loss: {:.4f} Acc: {:.4f}".format(val_loss_all[-1], val_acc_all[-1]))

        # 保存最高准确率的模型参数
        #验证集准确率最高时对应的模型状态字典
        if val_acc_all[-1] > best_acc:
            best_acc = val_acc_all[-1]
            best_model_wts = copy.deepcopy(model.state_dict())
        # 打印训练和验证的耗时
        time_use = time.time() - since
        print("time use: {:.0f}m {:.0f}s".format(time_use // 60, time_use % 60))

    # 选择最优的参数，保存最优参数的模型
    # torch.save()用于将Python对象序列化保存到文件
    torch.save(best_model_wts, 'best_model.pth')

    train_process = pd.DataFrame(data={"epoch": range(num_epochs),
                                       "train_loss_all": train_loss_all,
                                       "val_loss_all": val_loss_all,
                                       "train_acc_all": train_acc_all,
                                       "val_acc_all": val_acc_all})

    return train_process

def matplot_acc_loss(train_process):
    # 显示每一次迭代后的训练集和验证集的损失函数和准确率
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_process['epoch'], train_process.train_loss_all, "ro-", label="Train loss")
    plt.plot(train_process['epoch'], train_process.val_loss_all, "bs-", label="Val loss")
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("Loss")
    plt.subplot(1, 2, 2)
    plt.plot(train_process['epoch'], train_process.train_acc_all, "ro-", label="Train acc")
    plt.plot(train_process['epoch'], train_process.val_acc_all, "bs-", label="Val acc")
    plt.xlabel("epoch")
    plt.ylabel("acc")
    plt.legend()
    plt.show()

if __name__ == '__main__':
    #加载需要的模型
    LeNet = LeNet()
    #加载数据集
    train_dataloader, val_dataloader = train_val_data_process()
    #训练模型
    train_process = model_train_process(LeNet, train_dataloader, val_dataloader, num_epochs=20)
    matplot_acc_loss(train_process)






