import tensorflow as tf
import numpy as np
import model
import coding_matrix as cm





PATH = "Cleaned_5sRNA_test/"


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



