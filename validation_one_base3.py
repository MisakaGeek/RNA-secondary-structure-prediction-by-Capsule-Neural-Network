# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 15:27:45 2019

@author: 孙博文
计算一个碱基输入神经网络返回的判断结果，将其进行softmax处理并返回，用于test2.py
"""
import numpy as np
import tensorflow as tf
import model
import get_batch_data as gbd

ROW = 19
COLUMN = 128
BATCH_SIZE = 1
VEC_LEN = 8
logs_train_dir = "Net_model/" # 训练日志文件夹
def evaluate_one_base(train, train_label):
    """
    返回一个碱基的各部位对应(.)的概率
    input:
        train: 数据集，维度为[size,19,128,1]
        label: 标签集，还未经过ONE-HOT编码，维度为[size]
    outputs:
        prediction: 二维张量，第一维batch_size=1；第二维索引数3，对应该部位为（ . ）的概率
    """
    with tf.Graph().as_default():
        train_X, train_Y, one_hot_train_Y = gbd.get_batch_data(train, train_label, batch_size = BATCH_SIZE)
        train_logits, train_v_length = model.interface(inputs = train_X,
                                                Y = one_hot_train_Y,
                                                batch_size = BATCH_SIZE,
                                                vec_len = VEC_LEN,
                                                temp_batch_size = BATCH_SIZE)
        softmax = tf.nn.softmax(train_logits,dim=-1,name=None)   
        saver = tf.train.Saver(tf.global_variables())
        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state(logs_train_dir)
            if ckpt and ckpt.model_checkpoint_path:
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                saver.restore(sess,ckpt.model_checkpoint_path)
            else:
                print("no checkpoint file found")              
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess,coord = coord)
            try:
                if not coord.should_stop():
                    prediction = sess.run(softmax)
                    return prediction
            except Exception as e:
                coord.request_stop(e)
            finally:
                coord.request_stop()
                coord.join(threads)
