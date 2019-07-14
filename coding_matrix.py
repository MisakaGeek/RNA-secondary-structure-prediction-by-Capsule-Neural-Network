# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 20:16:47 2019
@author: 孙博文
"""
import math
import numpy as np
import random
import input_and_process_data as ipd
import data_same_size as dss
import tensorflow as tf






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
    result = coding_matrix(bases)  # ([['A','G','C','C','G','U']])
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





# 调用方法：variable_name = coding_matrix(bases)
"""运行方法：
# 导入包：
import input_and_process_data as ipd
import data_same_size as dss
import coding_matrix as cm

np.set_printoptions(threshold=np.inf, formatter={'float': '{:.1f}'.format}) 
# 这行的意义是输出时不丢失数据（无省略号），且规定输出的数据小数点后位数为1（改变主函数位置时别忘了也加上这一行）

# 下一行需要手动修改
PATH = "/"  # 规定.ct文件位置，注意最后应该有一个分隔符，因为内部调用时直接在PATH后append文件名，所以应在PATH里预先准备好分隔符

# 将ct文件1、2、5行读入 对应返回值三个列表
nums, bases, matches = ipd.Get_Batch_Data(PATH)
# 点括号序列转化 返回两个列表（此时stus是二维列表，是区分rna的）
bases, stus = ipd.Change_to_String(nums, bases, matches)

# label为最终输出的点括号序列列表
label = []
# 将label整合为一维列表，可以作为神经网络的标签集
for i in range(len(stus)):
    label = label + stus[i]

# result为从ct到最终统一大小的矩阵列表
result = []
# 转化为矩阵
result = cm.coding_matrix(bases)  # ([['A','G','C','C','G','U']])
# result[0][0]对应第一个rna的第一个矩阵 长度应该是rna的length
print(result[0][0])
print('\n')

# 调用滑窗算法，统一所有矩阵行数为19（超参数可修改）
result = dss.slide_window(result, 19)
# 为每个rna开头结尾各补了9行0 所以result[x][0~9]应该都是0行开头
print(result[0][10])
print('\n')

# 调用归一化，统一所有矩阵列数为128（超参数可修改）
result = dss.same_size(result, 128)
# 同上
print(result[0][10])

# 将inputs转化为np.array类型数据
inputs = np.array(result)
# 加上管道维度，inputs已经可以作为神经网络输入了
inputs = inputs.reshape(len(inputs), 19, 128, 1)
# 维度测试----预期：碱基数，19,128,1
print(inputs.shape)

# label已经可以作为神经网络的标签集了
label = np.array(label)
# 维度测试----预期：碱基数
print(label.shape)
"""

def coding_matrix(bases):
    """
    此过程形成许多个编码矩阵，每个编码矩阵对应一个ct文件的RNA序列，使用循环调用each_file_coding_matrix方法遍历所有ct文件
    bases是所有RNA序列所有信息，应为[[['A'],['G'],['C'],['C'],['G'],['U']],[['A'],['G'],['C'],['C'],['G'],['U']]]等类似格式
    """
    matrix=[]
    for each_base in bases:
        matrix.append(each_file_coding_matrix(each_base))
    return matrix

def each_file_coding_matrix(base):
    """
    此过程形成某个ct文件的编码矩阵，使用each_matrix_value子过程计算编码矩阵的具体数值
    base是一条RNA序列所有信息，应为[['A'],['G'],['C'],['C'],['G']]等类似格式
    """
    length = len(base)  # RNA序列长度
    array = np.zeros((length, length))
    line = 0  # 行号
    while line < length:
        col = 0  # 列号
        while col < length:
            array[line][col] = each_matrix_value(base, line, col)
            col += 1
        line += 1
    return array

def each_matrix_value(base, i, j):
    """
    此过程最终形成矩阵上第i行第j列的数值，需要子过程init_value辅助计算权值
    base是一条RNA序列所有信息，应为[['A'],['G'],['C'],['C'],['G']]等类似格式
    i和j是两个整数，代表碱基在RNA序列base中的位置下标
    """
    weight = init_value(base, i, j)  # 编码矩阵第i行第j列的值，初始为0，如果配对，则赋值为配对权值
    if weight != 0:
        auxi_weight = 0  # 两侧碱基配对情况对当前配对碱基的影响值
        p = 1  # 迭代计数器
        score = init_value(base, i-p, j+p)
        while score != 0:  # 只要左右两侧碱基配对，就一直计算影响值
            auxi_weight += score * math.exp(-0.5*p*p)
            p += 1
            score = init_value(base, i-p, j+p)
        weight += auxi_weight
        # 开始另一方向的计算
        auxi_weight = 0
        p = 1  # 重置影响值与迭代计数器
        score = init_value(base, i+p, j-p)
        while score != 0:
            auxi_weight += score * math.exp(-0.5*p*p)
            p += 1
            score = init_value(base, i+p, j-p)
        weight += auxi_weight
    return weight

def init_value(base, i, j, U_G_Weight = 0.8):
    """
    i和j是两个整数，代表碱基在RNA序列base中的位置下标(01234...)
    此过程通过读取一条RNA序列信息base，判断位置为i和j的两碱基初始编码值
    U_G_Weight为U-G配对的权值，根据实验结果，设为0.8时表现最好，这里设为默认参数
    """
    if (i < 0 or j < 0 or i >= len(base) or j >= len(base)):
        return 0
    else:
        if (base[i] == ['A'] and base[j] == ['U']) or (base[i] == ['U'] and base[j] == ['A']):  # AU配对为2
            return 2
        elif (base[i] == ['C'] and base[j] == ['G']) or (base[i] == ['G'] and base[j] == ['C']):  # CG配对为3
            return 3
        elif (base[i] == ['U'] and base[j] == ['G']) or (base[i] == ['G'] and base[j] == ['U']):  # UG配对为0到2随机实数
            return U_G_Weight
        else:
            return 0




if __name__ == "__main__":
    PATH = "Cleaned_5sRNA_test/"
    row = 19
    column = 128
    train_X, train_Y=get_Data(PATH,row,column)
