# -*- coding: utf-8 -*-
"""
Created on Sun Jul 14 22:20:27 2019

@author: 孙博文
验证集
"""

import numpy as np
import tensorflow as tf
import model
import get_batch_data as gbd

ROW = 19
COLUMN = 128
BATCH_SIZE = 1
VEC_LEN = 8

def evaluate():
    with tf.Graph().as_default():
        # 训练日志文件夹
        logs_train_dir = '/'
        # 验证文件夹
        verification_dir = "/"
        # 数据总数
        n_test = 0
        train, train_label = gbd.get_Data(verification_dir, ROW, COLUMN)
        train_X, train_Y, one_hot_train_Y = gbd.get_batch_data(train, train_label, batch_size = BATCH_SIZE)
        train_logits, train_v_length = model.interface(inputs = train_X,
                                                Y = one_hot_train_Y,
                                                batch_size = BATCH_SIZE,
                                                vec_len = VEC_LEN,
                                                temp_batch_size = BATCH_SIZE)
        top_k_op = tf.nn.in_top_k(train_logits, train_Y, 1)       
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
                num_iter = int(n_test/BATCH_SIZE)
                true_count = 0
                total_sample_count = num_iter * BATCH_SIZE
                step = 0
                while step < num_iter and not coord.should_stop():
                    prediction = sess.run([top_k_op])
                    true_count += np.sum(prediction)
                    step += 1
                    precision = float(true_count)/total_sample_count
                print("precision = %3f"%precision)
            except Exception as e:
                coord.request_stop(e)
            finally:
                coord.request_stop()
                coord.join(threads)

if __name__ == "__main__":
    evaluate()
