# -*- coding: utf-8 -*-
"""
Created on Thu May 16 18:20:27 2019

@author: 雷浩洁
重新写胶囊神经网络架构
不再使用类的方式，直接用函数
"""

import numpy as np
import tensorflow as tf

def squash(vector):
    """
    压缩操作，将一个向量的长度压缩到1以内，并不损坏其特征
    参数:
        input:一个维度为[batch_size, 1, num_caps, vec_len, 1]
                      或[batch_size, num_caps, vec_len, 1]的张量
    返回:
        一个形状相同但第三，第四维被压缩的向量
    """
    vec_squared_norm = tf.reduce_sum(tf.square(vector),
                                     -2,
                                     keep_dims=True)
    scalar_factor = vec_squared_norm / (1 + vec_squared_norm) / tf.sqrt(vec_squared_norm + 1e-9)
    vec_squashed = scalar_factor * vector  # element-wise
    return(vec_squashed)
    
def get_shape(inputs,name=None):
    name="shape" if name is None else name
    name = "shape" if name is None else name
    with tf.name_scope(name):
        static_shape = inputs.get_shape().as_list()
        dynamic_shape = tf.shape(inputs)
        shape = []
        for i, dim in enumerate(static_shape):
            dim = dim if dim is not None else dynamic_shape[i]
            shape.append(dim)
        return(shape)
        
def routing(input,b_IJ,num_outputs=3,num_dims=4):
    """
    路由算法
    参数:
        input:一个形状为[batch_size, num_caps_l=14880, 1, len_vec=2, 1]的5维张量
        num_outputs:输出胶囊的数量，因为要预测的结果只有三种可能，即左括号，右括号或点，因此是3个
        num_dims:输出向量的维度
    返回:
        一个形状为[batch_size, num_caps_l_plus_1, vec_len=2, 1]的4维张量
        
    u_i表示第l层的第i个胶囊的输出向量
    v_j表示第l+1层第j个胶囊的输出向量
    参数W的维度:[1, num_caps_i, num_caps_j*len_v_j, len_u_j, 1]
    """
    input_shape = input.get_shape()

    W=tf.get_variable('Weight',
                      shape=[1, input_shape[1], num_dims*num_outputs, 2, 1],
                      dtype=tf.float32,
                      initializer=tf.random_normal_initializer())
    biases=tf.get_variable('bias',
                           shape=[1,1,num_outputs,num_dims,1])
          
    input=tf.tile(input,
                  [1,1,num_dims*num_outputs,1,1])
    
    u_hat = tf.reduce_sum(W*input,
                          axis=3,
                          keep_dims=True)
    u_hat = tf.reshape(u_hat,
                       shape=[-1,input_shape[1],num_outputs,num_dims,1])

    u_hat_stopped=tf.stop_gradient(u_hat,
                                   name='stop_gradient')
    
    for r_iter in range(1):
        with tf.variable_scope('iter_'+str(r_iter)):
            c_IJ=tf.nn.softmax(b_IJ,
                               axis=2)
            
            if r_iter==2:
                s_J=tf.multiply(c_IJ,
                                u_hat)
                
                s_J=tf.reduce_sum(s_J,
                                  axis=1,
                                  keep_dims=True)+biases
                v_J=squash(s_J)
            elif r_iter<2:
                s_J=tf.multiply(c_IJ,
                                u_hat_stopped)
                s_J=tf.reduce_sum(s_J,
                                  axis=1,
                                  keep_dims=True)+biases
                v_J=squash(s_J)
                
                v_J_tiled=tf.tile(v_J,
                                  [1,input_shape[1],1,1,1])
                u_produce_v=tf.reduce_sum(u_hat_stopped*v_J_tiled,
                                          axis=3,
                                          keep_dims=True)
                b_IJ+=u_produce_v

    return (v_J) 
        
def interface(inputs,Y,batch_size,vec_len,temp_batch_size):
    """
    定义神经网络模型
    Args:
        inputs:输入数据，一个维度为[batch_size,height,width,channels]的4维张量
               其中对于5sRNA，height=19,width=128,channels=1
        Y:经过独热码编码过的标签集，是一个维度为[batch_size,3]的矩阵
        batch_size:一次训练的数据集大小
        num_outputs:输出可划分为几类，本项目为3
        vec_len:向量的长度,2或是4
        temp_batch_size:数据集大小
    Return:
        logits:输出的预测结果，是一个[batch_size,num_outputs]的张量
    """
    num_outputs = 3
    # 输出可划分为几类，本项目为3
    
    # graph=tf.Graph()
    # 通过tensorboard用图形化界面展示出来流程结构
    # 目前暂时用不着，之后补充
    
    with tf.variable_scope('Conv1_layer'):
        #第一层为卷积层，使用大小为[3,3,8]的卷积核进行卷积操作
        #维度变化为[batch_size,19,128,1] --> [batch_size,17,126,8]
        conv1=tf.contrib.layers.conv2d(inputs,
                                       num_outputs=8,
                                       kernel_size=3,
                                       stride=1,
                                       padding='VALID')
        assert(conv1.shape == [temp_batch_size, 17, 126, 8])
    
    with tf.variable_scope('PrimaryCaps_layer'):
        # 第二层为主胶囊层，使用8个胶囊，向量长度为2，不使用动态路由算法
        # 维度变化为[batch_size,15,126,8] --> [batch_size,14880,2,1]
        num_outputs2=8  # 本层共有8个胶囊用以输出
        vec_len2=2 # 本层向量长度为2
        kernel_size2=3  # 本层卷积核大小为3
        caps1=tf.contrib.layers.conv2d(conv1,
                                       num_outputs2*vec_len2,
                                       kernel_size2,
                                       stride=1,
                                       padding="VALID",
                                       activation_fn=tf.nn.relu)
        # 改变维度，将每一个矩阵拉伸为14880个2维向量
        # (14880=15*124*8)
        caps1=tf.reshape(caps1,
                         shape=[temp_batch_size,-1,vec_len2,1])
        assert(caps1.shape==[temp_batch_size,14880,2,1])
    
    with tf.variable_scope('DigitCaps_layer'):
        # 第三层，数字胶囊层，可理解为向量版的全连接层，本层共有3个胶囊
        # 使用动态路由算法进行迭代训练
        # 维度变化为[batch_size,14880,2,1] --> [batch_size,3,4,1]
        
        vec_len3=4  # 本层向量长度为8
        digitcaps=tf.reshape(caps1,
                             shape=(temp_batch_size,-1,1,caps1.shape[-2].value,1))
        # [batch_size,14880,2,1] --> [batch_size,14880,1,2,1]
        
        with tf.variable_scope('routing'):
            # b_IJ的维度为[batch_size, num_caps_l, num_caps_l_plus_1, 1, 1]

            b_IJ = tf.constant(np.zeros([temp_batch_size, digitcaps.shape[1].value, num_outputs, 1, 1], dtype=np.float32))

            digitcaps=routing(digitcaps,
                              b_IJ,
                              num_outputs=num_outputs,
                              num_dims=vec_len3)
            digitcaps = tf.squeeze(digitcaps, axis=1)
            
    with tf.variable_scope('Masking'):
            #计算张量长度，并使用softmax
            #维度变化:[batch_size,3,4,1]->[batch_size,3,1,1]
            v_length=tf.sqrt(tf.reduce_sum(tf.square(digitcaps),
                                                axis=2,
                                                keep_dims=True)+1e-9)
            softmax_v=tf.nn.softmax(v_length,
                                    axis=1)
            assert softmax_v.get_shape()==[temp_batch_size,num_outputs,1,1]
            
            #找到3个胶囊输出的向量中长度最大的向量的索引
            #维度变化为:[batch_size,3,1,1]->[batch_size](index)
            argmax_index=tf.to_int32(tf.argmax(softmax_v, axis=1))
            assert argmax_index.get_shape()==[temp_batch_size,1,1]
            argmax_index=tf.reshape(argmax_index,
                                    shape=(temp_batch_size,))
            
            # 除了特定的 Capsule 输出向量，需要蒙住(masking)其它所有的输出向量
            #mask_with_y是否用真实标签蒙住目标Capsule
            mask_with_y=True
            if mask_with_y==True:
                #这一步有问题，对于标签Y的维度还应再讨论一下
                #应该是还需要把Y变成独热码才能用
                masked_v=tf.multiply(tf.squeeze(digitcaps),tf.reshape(Y,(-1,3,1)))
                
                v_length=tf.sqrt(tf.reduce_mean(tf.square(digitcaps),axis=2,keep_dims=True)+1e-9)
                              
    #全连接层解码部分,暂时添上，看看效果怎么样,以后可能会删掉
    #3层全连接层，最后一层为三个输出的全连接层
    #[batch_size,1,8,1]=>[batch_size,8]=>[batch_size,16]
    with tf.variable_scope('Decoder'):
        vector_j=tf.reshape(masked_v,shape=(temp_batch_size,-1))
        fc1=tf.contrib.layers.fully_connected(vector_j,
                                              num_outputs=16)
        fc2=tf.contrib.layers.fully_connected(fc1,
                                              num_outputs=16)
        logits=tf.contrib.layers.fully_connected(fc2,
                                                 num_outputs=num_outputs,
                                                 activation_fn=tf.sigmoid)
    return logits, v_length

def loss(logits,v_length,labels,Y,temp_batch_size):
    """
    #定义损失函数
    Args:
        logits:神经网络的预测结果，一个维度为[batch_size,num_outputs]的矩阵
        v_length:向量的模长
        labels:未经过独热码处理的数据标签集，一个维度为[batch_size]的一位序列
        Y:经过独热码编码过的标签集，是一个维度为[batch_size,3]的矩阵
        temp_batch_size:数据集个数
    Return:
        loss:损失值
    """
    #[batch_size,3,1,1]
    #1.(margin loss)
    # max_l = max(0, m_plus-||v_c||)^2
    m_plus=0.9
    max_l = tf.square(tf.maximum(0., m_plus - v_length)) 
    
    # max_r = max(0, ||v_c||-m_minus)^2
    m_minus=0.1
    max_r = tf.square(tf.maximum(0., v_length -m_minus))
    assert max_l.get_shape()== [temp_batch_size,3,1,1]
    
    #reshape:[batch_size,3,1,1] -> [batch_size,3]
    max_l = tf.reshape(max_l,
                       shape=(temp_batch_size, -1))
    max_r = tf.reshape(max_r,
                       shape=(temp_batch_size, -1))
    
    #计算T_c:[batch_size,3]
    #有人说T_c=Y,但是我感觉有些不对劲，此处存疑，暂且按T_c=Y处理
    # #Y目前是用独热码编码过的，因此有一定道理
    T_c=Y
    lambda_val=0.5
    L_c=T_c*max_l+lambda_val*(1-T_c)*max_r
        
    margin_loss=tf.reduce_mean(tf.reduce_sum(L_c,axis=1))
    
    # 2.(The reconstruction loss)
    # 不能照搬Hinton使用的解码器的loss函数
    # 原因是解码器的loss函数原理是计算生成图片与原图片的像素“距离”，784个像素点的值对应地进行计算
    # 而这边的预测结果只有三个神经元，输入矩阵却有11*120个，根本无法计算
    # 因此在这里先使用tf.nn.sparse_softmax_cross_entropy_with_logits进行计算
    # 不知效果如何，有带改进
    
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                                   labels=labels,
                                                                   name='xentropy_per_example')
    reconstruction_loss = tf.reduce_mean(cross_entropy,name = 'reconstruction_loss')
    
    regularization_scale=0.392#正则化参数
    #总损失函数
    loss=margin_loss+regularization_scale*reconstruction_loss
    
    return loss

def trainning(loss,learning_rate):
    """
    进行一步训练(即优化)操作
    Args:
        loss:计算出的loss函数
        learning_rate:学习率
    Return:
        train_op:一步优化操作
    """
    with tf.variable_scope('Optimizer'):
        optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
        global_step = tf.Variable(0,name = 'global_step',trainable = False)
        train_op = optimizer.minimize(loss,global_step = global_step)
    
    return train_op

def evalution(logits,labels):
    """
    在训练过程中，对模型预测的准确率进行评估
    以实时监测训练效果
    Args:
        logits:神经网络预测结果
        labels:未经过独热码处理的数据标签集，一个维度为[batch_size]的一位序列
    Return:
        accuracy:计算出的预测准确率
    """
    with tf.variable_scope('accuracy'):
        correct=tf.nn.in_top_k(logits,
                               labels,
                               1)
        correct= tf.cast(correct,
                         tf.float32)
        accuracy = tf.reduce_mean(correct)
        #tf.summary.scalar(scope.name + '/accuracy',accuracy)
    return accuracy
