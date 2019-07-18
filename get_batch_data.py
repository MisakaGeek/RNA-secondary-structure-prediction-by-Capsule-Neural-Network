# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 14:27:23 2019
@author: 雷浩洁
"""
import tensorflow as tf
import numpy as np
import model
import coding_matrix as cm
import input_and_process_data as ipd
import data_same_size as dss

PATH = "Cleaned_5sRNA_test/"

def get_Data(PATH, row=19, column=128):
    """
    获取数据集
    将数据预处理部分的代码整合到一起
    只需调用这个函数，输入路径与想要设置的矩阵大小，就能返回符合要求的，所有的数据集
    :param PATH:读取的数据文件路径名
    :param row:滑窗大小，控制矩阵的行值
    :param column:归一化大小，控制矩阵列值
    :return:inputs:所有数据集，维度为[size,19,128,1]
            labels: 所有标签集，还未经过ONE-HOT编码，维度为[size]
    """
    # 将ct文件1、2、5行读入 对应返回值三个列表
    nums, bases1, matches = ipd.Get_Batch_Data(PATH)
    # 点括号序列转化 返回两个列表（此时stus是二维列表，是区分rna的）
    bases, stus = ipd.Change_to_String(nums, bases1, matches)
    print("读入数据完成！")

    # 整合标签集
    # label为最终输出的点括号序列列表
    label = []
    # 将label整合为一维列表，可以作为神经网络的标签集
    for i in range(len(stus)):
        label = label + stus[i]
    label = np.array(label)
    labels= []
    for i in range(len(label)):
        if label[i]=='(':
            labels.append(0)  # 将 ( 编码为0
        elif label[i] == '.':
            labels.append(1)  # 将 . 编码为1
        elif label[i] == ')':
            labels.append(2)  # 将 ) 编码为2
    print("转化标签完成！")

    # result为从ct到最终统一大小的矩阵列表
    result = []
    # 转化为矩阵
    result = cm.coding_matrix(bases)  # ([['A','G','C','C','G','U']])
    # result[0][0]对应第一个rna的第一个矩阵 长度应该是rna的length
    print("矩阵转换完成！")

    # 调用滑窗算法，统一所有矩阵行数为19（超参数可修改）
    result = dss.slide_window(result, row=row)
    # 为每个rna开头结尾各补了9行0 所以result[x][0~9]应该都是0行开头
    print("滑窗算法完成！")



    # 调用归一化，统一所有矩阵列数为128（超参数可修改）
    result = dss.same_size(result, column=column)
    print("归一化完成！")

    # 将inputs转化为np.array类型数据
    inputs = np.array(result)
    # 加上管道维度，inputs已经可以作为神经网络输入了
    inputs = inputs.reshape(len(inputs), row, column, 1)
    assert (inputs.shape==(len(inputs), row, column, 1))

    return inputs, labels


def get_batch_data(images, labels, batch_size=256):
    """
    对得来的数据进行划分分批操作，同时对标签进行onehot编码，一并输出。
    input:
        images:4-D array, train set
        labels:1-D array, the label of train set
        batch_size: how large you set your batch
    outputs:
        image_batch: which shape is [batch_size, row, column, channel]
        label_batch: which shape is [batch_size, ]
        one_hot_label_batch: which shape is [batch_size, 3]
    """

    # 数据类型转换为tf.float32
    images = tf.cast(images, tf.float32)
    labels = tf.cast(labels, tf.int32)

    # 从tensor列表中按顺序或随机抽取一个tensor准备放入文件名称队列
    input_queue = tf.train.slice_input_producer([images, labels], shuffle=True)

    # 从文件名称队列中读取文件准备放入文件队列
    image_batch, label_batch = tf.train.batch(input_queue, batch_size=batch_size, num_threads=2, capacity=64,
                                              allow_smaller_final_batch=False)
    one_hot_label_batch = tf.one_hot(label_batch,
                                depth=3,
                                axis=1,
                                dtype=tf.float32)  # 转化为one-hot编码
    return image_batch, label_batch, one_hot_label_batch
