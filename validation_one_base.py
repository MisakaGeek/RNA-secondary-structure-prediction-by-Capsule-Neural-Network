# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 15:27:45 2019

@author: 孙博文
计算一个ct文件输入神经网络返回的判断结果，将其进行softmax处理并返回，用于test.py
"""
import numpy as np
import tensorflow as tf
import model
import get_batch_data as gbd

ROW = 19
COLUMN = 128
BATCH_SIZE = 1
VEC_LEN = 8
logs_train_dir = "" # 训练日志文件夹

def evaluate_one_base(ct):
    """
    返回一条碱基的各部位对应(.)的概率
    input:
        ct: 文件夹，内部应只有一个ct文件，多了后果自负
    outputs:
        prediction: 二维张量，第一维因为batch_size = 1，一般索引数为碱基数；第二维索引数3，对应该部位为（ . ）的概率
    """
    with tf.Graph().as_default():
        train, train_label = gbd.get_Data(ct, ROW, COLUMN)
        train_X, train_Y, one_hot_train_Y = gbd.get_batch_data(train, train_label, batch_size = BATCH_SIZE)
        train_logits, train_v_length = model.interface(inputs = train_X,
                                                Y = one_hot_train_Y,
                                                batch_size = BATCH_SIZE,
                                                vec_len = VEC_LEN,
                                                temp_batch_size = BATCH_SIZE)
        softmax = tf.nn.softmax(train_logits,dim=-1,name=None)   
        saver = tf.train.Saver(tf.global_variables())
        with tf.Session() as sess:
            print("Reading checkpoints...")
            ckpt = tf.train.get_checkpoint_state(logs_train_dir)
            if ckpt and ckpt.model_checkpoint_path:
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                saver.restore(sess,ckpt.model_checkpoint_path)
                print("Loading success,global_step is %s" % global_step)
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
