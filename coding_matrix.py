# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 20:16:47 2019
@author: 孙博文
"""

import math
import numpy as np
import random
import input_and_process_data as ipd

# 调用方法：variable_name = code_matrices(nums, bases, matches)
"""运行方法:
PATH = "./src/"  # 规定.ct文件位置
nums, bases, matches = ipd.Get_Batch_Data(PATH)  # 获得RNA序列信息
result = code_matrices(nums, bases, matches)  # 获得矩阵
print(result)  # result即可作为PATH文件夹下所有RNA序列的编码矩阵
"""
def code_matrices(nums, bases, matches):
    """
    nums为所有RNA序列的下标信息(1,2,...)
    bases为所有RNA序列的碱基名字序列(AGCU)
    matches为所有RNA序列的配对碱基的下标(0,1,2,...)
    而rna是把上面三个给zip到了一起，代表一条RNA所有信息
    此过程最终形成矩阵list，使用each_code_matrix子过程计算每条RNA序列的编码矩阵
    """
    matrices = []  # 所有RNA序列编码矩阵的列表
    for each_num, each_base, each_match in list(zip(nums, bases, matches)):  # 秘技：三重并行迭代
        matrices.append(each_matrix(list(zip(each_num, each_base, each_match))))
    return matrices


#  所有rna变量均为zip后的tuple类型，是把num,base,match三个信息统合后形成的tuple
def each_matrix(rna):
    """
    此过程最终形成矩阵，使用each_matrix_value子过程计算一个编码矩阵的具体数值
    """
    length = len(list(rna))  # RNA序列长度
    array = np.zeros((length, length))
    line = 0  # 行号
    while line < length:
        col = 0  # 列号
        while col < length:
            array[line][col] = each_matrix_value(rna, line+1, col+1)  # 碱基位置下标从1开始，而矩阵编号从0开始，因此需要加1
            col += 1
        line += 1
    return array


def each_matrix_value(rna, i, j):
    """
    rna是一条RNA序列所有信息
    i和j是两个整数，代表碱基在RNA序列rna中的位置下标
    此过程最终形成矩阵上第i行第j列的数值，需要两个子过程is_paired和init_value辅助
    """
    weight = 0.0  # 编码矩阵第i行第j列的值，初始为0
    if is_paired(rna, i, j):
        weight = init_value(rna, i, j)  # 如果配对，则赋值为配对权值
        auxi_weight = 0.0  # 两侧碱基配对情况对当前配对碱基的影响值
        p = 1  # 迭代计数器
        while is_paired(rna, i-p, j+p):  # 只要左右两侧碱基配对，就一直计算影响值
            auxi_weight += init_value(rna, i-p, j+p) * math.exp(-(p*p)/2)
            p += 1
        weight += auxi_weight
        auxi_weight = 0.0
        p = 1  # 重置影响值与迭代计数器
        while is_paired(rna, i+p, j-p):
            auxi_weight += init_value(rna, i+p, j-p) * math.exp(-(p*p)/2)
            p += 1
        weight += auxi_weight
    return weight
	
	
def is_paired(rna, i, j):
    """
    i和j是两个整数，代表碱基在RNA序列rna中的位置下标，从1开始
    此过程通过读取一条RNA序列完整信息rna，判断位置为i和j的两碱基是否配对
    """
    length = len(list(rna))
    if i < 1 or i > length or j < 1 or j > length:  # 下标越界，不匹配
        return False
    else:
        if int(rna[i-1][0]) == i and int(rna[j-1][0]) == j and int(rna[i-1][2]) == j:  # 计算是否匹配；由于从0开始计数，所以i和j都得减1
            # 对于rna[i-1]，它是一个list，第一个元素为num，即位置下标序号；第二个元素为base，即碱基信息（AUCG中的一个）；第三个元素为match，即配对碱基的位置下标序号，未配对为0
            return True
        else:
            return False


def init_value(rna, i, j):
    """
    i和j是两个整数，代表碱基在RNA序列rna中的位置下标
    此过程通过读取一条RNA序列完整信息rna，判断位置为i和j的两碱基初始编码值
    """
    if not is_paired(rna, i, j):  # 不配对，直接返回0
        return 0
    else:
        if (rna[i-1][1] == 'A' and rna[j-1][1] == 'U') or (rna[i-1][1] == 'U' and rna[j-1][1] == 'A'):  # AU配对为2
            return 2
        elif (rna[i-1][1] == 'C' and rna[j-1][1] == 'G') or (rna[i-1][1] == 'G' and rna[j-1][1] == 'C'):  # CG配对为3
            return 3
        elif (rna[i-1][1] == 'U' and rna[j-1][1] == 'G') or (rna[i-1][1] == 'G' and rna[j-1][1] == 'U'):  # UG配对为0到2随机实数
            return random.uniform(0, 2)
        else:
            return 0
