# -*- coding: utf-8 -*-
"""
Created on Sun Jul 14 22:21:17 2019

@author: 孙博文
测试集
"""
import validation_one_base as vob
import input_and_process_data as ipd
import coding_matrix as cm
import data_same_size as dss
import pandas as pd
import numpy as np
import tensorflow as tf
import os
import codecs

ROW = 19
COLUMN = 128
test_data_path = "test" # 存放测试集ct文件的文件夹，应有许多子文件夹，以保证每个子文件夹内只有一个ct文件
'''
Example: test --- 5s_Acetobacter-sp.-1 --- 5s_Acetobacter-sp.-1.ct
              --- 5s_Acidithiobacillus-ferrooxidans-2 --- 5s_Acidithiobacillus-ferrooxidans-2.ct
              ...
'''
# 最后不要加文件分隔符
testDir = os.listdir(test_data_path)
"""
流程：读取含有一个ct文件的文件夹，将这个ct文件输入神经网络，获取RNA二级结构预测结果，在对应文件夹内输出prediction.csv文件，即为二级结构预测结果
"""

name = ['left','point','right']
'''
# 初版：设置batch_size为ct文件的长度
for ct in testDir: # ct是文件夹，内部有且仅有一个.ct格式的碱基文件
    work_dir = test_data_path + '/' + ct + '/'
    for file_name in os.listdir(work_dir):
        f = codecs.open(work_dir + file_name, mode = 'r')
        headline = f.readline() # 标题行，写入每一个文件中
        length = int(headline.split()[0])
    result = vob.evaluate_one_base(work_dir, length)
    prediction = pd.DataFrame(columns = name, data = result)
    prediction.to_csv(work_dir + "test.csv") # prediction.csv保存各个位置是左括号，右括号，点的概率
'''
'''
# 第二版：为每个碱基建立新文件夹和专属ct文件，奇慢无比
for ct in testDir:
    work_dir = test_data_path + '/' + ct + '/'
    for file_name in os.listdir(work_dir):
        f = codecs.open(work_dir + file_name, mode = 'r')
        headline = f.readline() # 标题行，写入每一个文件中
    nums, bases, matches = ipd.Get_Batch_Data(work_dir)
    result_list = [] # n*3大小的二维list，n行代表输入的ct文件有n个碱基，3代表每个碱基应该是左括号，右括号，点的概率
    for index in nums[0]: # nums为二维数组, i为一维数组
        os.mkdir(work_dir + index)
        each_file = open(work_dir + index + '/' + index + '.ct', 'w')
        each_file.write(headline)
        each_file.write(nums[0][int(index)-1])
        each_file.write(' ')
        each_file.write(bases[0][int(index)-1])
        each_file.write(' 1 2 ') # 第三列和第四列
        each_file.write(matches[0][int(index)-1])
        each_file.close()
        temp_result = vob.evaluate_one_base(work_dir + index + "/")
        result = temp_result[0][0], temp_result[0][1], temp_result[0][2]
        result_list.append(result)
        os.remove(work_dir + index + '/' + index + '.ct')
        os.rmdir(work_dir + index)
    prediction = pd.DataFrame(columns = name, data = result_list)
    prediction.to_csv(work_dir + "prediction.csv")
'''

# 第三版：将一个ct文件拆分，仿佛其是由多个只含有一个碱基的ct文件得来（实际上从同一文件读入，节约了文件操作时间）
for ct in testDir:
    result_list = [] # n*3大小的二维list，n行代表输入的ct文件有n个碱基，3代表每个碱基应该是左括号，右括号，点的概率
    work_dir = test_data_path + '/' + ct + '/'
    nums_data, bases_data, matches_data = ipd.Get_Batch_Data(work_dir)
    nums_split = []
    bases_split = []
    matches_split = []
    for i in nums_data[0]:
        temp_i = []
        temp_i.append(i)
        nums_split.append(temp_i)
    for j in bases_data[0]:    
        temp_j = []
        temp_j.append(j)
        bases_split.append(temp_j)
    for k in matches_data[0]:
        temp_k = []
        temp_k.append(k)
        matches_split.append(temp_k)
    for each_base in zip(nums_split, bases_split, matches_split):
        nums = each_base[0]
        bases = each_base[1]
        matches = each_base[2]
        bases, stus = ipd.Change_to_String(nums, bases, matches)
        label = []
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
        result = []
        result = cm.coding_matrix(bases)
        result = dss.slide_window(result, row = ROW)
        result = dss.same_size(result, column = COLUMN)
        inputs = np.array(result)
        inputs = inputs.reshape(len(inputs), ROW, COLUMN, 1)
        temp_result = vob.evaluate_one_base(inputs, labels)
        result = temp_result[0][0], temp_result[0][1], temp_result[0][2]
        result_list.append(result)
    prediction = pd.DataFrame(columns = name, data = result_list)
    prediction.to_csv(work_dir + "prediction.csv")
