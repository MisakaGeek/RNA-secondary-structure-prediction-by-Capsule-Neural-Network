# -*- coding: utf-8 -*-
"""
Created on Sun Jul 14 22:21:17 2019

@author: 孙博文
测试集
"""
import validation_one_base as vob
import pandas as pd
import os

result_list = [] # n*3大小的二维list，n行代表输入的ct文件有n个碱基，3代表每个碱基应该是左括号，右括号，点的概率
test_data_path = '/' # 存放测试集ct文件的文件夹，应有许多子文件夹，以保证每个子文件夹内只有一个ct文件

testDir = os.listdir(test_data_path)
testDir.sort()
"""
流程：读取含有一个ct文件的文件夹，将这个ct文件输入神经网络，获取RAN二级结构预测结果，在对应文件夹内输出两个csv文件
"""
for ct in testDir: # ct是文件夹，内部有且仅有一个.ct格式的碱基文件
    result_list = vob.evaluate_one_base(ct)
    name = ['left','right','point']
    possibility = pd.DataFrame(columns = name, data = result_list)
    possibility.to_csv(os.getcwd() + '\\' + ct + "\\possibility.csv") # possibility.csv保存各个位置是左括号，右括号，点的概率
    result = []
    for index, row in possibility.iterrows():
        if row["left"] >= row["right"] and row["left"] >= row["point"]:
            result.append(['('])
        elif row["right"] >= row["left"] and row["right"] >= row["point"]:
            result.append([')'])
        elif row["point"] >= row["left"] and row["point"] >= row["right"]:
            result.append(['.'])
    result_csv = pd.DataFrame(data = result, columns = ["result"])
    result_csv.to_csv(os.getcwd() + '\\' + ct + "\\result.csv") # result.csv保存各个位置是左括号，右括号或者点
