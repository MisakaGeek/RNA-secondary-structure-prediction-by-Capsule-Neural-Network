# -*- coding: utf-8 -*-
"""
Created on Sun Jul 14 22:21:17 2019

@author: 孙博文
测试集
第三版：将一个ct文件拆分，仿佛其是由多个只含有一个碱基的ct文件得来（实际上从同一文件读入，节约了文件操作时间），但是准确率低于第二版
"""
import validation_one_base3 as vob3
import input_and_process_data as ipd
import coding_matrix as cm
import data_same_size as dss
import final_process as fp
import pandas as pd
import numpy as np
import os

ROW = 19
COLUMN = 128
test_data_path = "/" # 存放测试集ct文件的文件夹，应有许多子文件夹，以保证每个子文件夹内只有一个ct文件
'''
Example: test --- 5s_Acetobacter-sp.-1 --- 5s_Acetobacter-sp.-1.ct
              --- 5s_Acidithiobacillus-ferrooxidans-2 --- 5s_Acidithiobacillus-ferrooxidans-2.ct
              ...
'''
"""
流程：读取含有一个ct文件的文件夹，将这个ct文件输入神经网络，获取RNA二级结构预测结果，在对应文件夹内输出prediction.csv文件，即为二级结构预测结果
"""
testDir = os.listdir(test_data_path)
name = ['left','point','right']
for ct in testDir:
    result_list = [] # n*3大小的二维list，n行代表输入的ct文件有n个碱基，3代表每个碱基应该是左括号，右括号，点的概率
    work_dir = test_data_path + ct + '/'
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
        temp_result = vob3.evaluate_one_base(inputs, labels)
        result = temp_result[0][0], temp_result[0][1], temp_result[0][2]
        result_list.append(result)
    prediction = pd.DataFrame(columns = name, data = result_list) # 输出文件，若只想看评测参数请注释掉本行和下一行
    prediction.to_csv(work_dir + "prediction.csv")
    final_pre = fp.Nus_p(result_list, bases_data[0])
    pre_match = fp.change_to_match(final_pre)
    match = []
    for mat in matches_data[0]:
        match.append(int(mat))
    TP, FN, FP, R, P, F1 = fp.estimate(pre_match, match)
    print('查全率：' + str(R))
    print('查准率：' + str(P))
    print('综合衡量：' + str(F1))
    print("将在浏览器中显示图形结果，请等待")
    fp.open_in_webbrowser(final_pre, bases[0]) # 开启默认浏览器
