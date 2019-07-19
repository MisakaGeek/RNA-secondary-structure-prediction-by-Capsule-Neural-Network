# -*- coding: utf-8 -*-
"""
Created on Sun Jul 14 22:21:17 2019

@author: 孙博文
测试集
"""
import validation_one_base as vob
import pandas as pd
import os

test_data_path = "" # 存放测试集ct文件的文件夹，应有许多子文件夹，以保证每个子文件夹内只有一个ct文件
# 最后不要加文件分隔符
testDir = os.listdir(test_data_path)
"""
流程：读取含有一个ct文件的文件夹，将这个ct文件输入神经网络，获取RAN二级结构预测结果，在对应文件夹内输出两个csv文件
"""
result_list = [] # n*3大小的二维list，n行代表输入的ct文件有n个碱基，3代表每个碱基应该是左括号，右括号，点的概率
name = ['left','right','point']
for ct in testDir: # ct是文件夹，内部有且仅有一个.ct格式的碱基文件
    result_list = vob.evaluate_one_base(ct)
    prediction = pd.DataFrame(columns = name, data = result_list)
    prediction.to_csv(test_data_path + '\\' + ct + "\\prediction.csv") # prediction.csv保存各个位置是左括号，右括号，点的概率
