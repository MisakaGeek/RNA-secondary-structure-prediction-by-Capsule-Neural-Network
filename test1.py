# -*- coding: utf-8 -*-
"""
Created on Sun Jul 14 22:21:17 2019

@author: 孙博文
测试集
第一版：设置batch_size为ct文件的长度
"""
import validation_one_base1 as vob1
import input_and_process_data as ipd
import final_process as fp
import pandas as pd
import os
import codecs

test_data_path = "test/" # 存放测试集ct文件的文件夹，应有许多子文件夹，以保证每个子文件夹内只有一个ct文件
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
for ct in testDir: # ct是文件夹，内部有且仅有一个.ct格式的碱基文件
    work_dir = test_data_path + ct + '/'
    for file_name in os.listdir(work_dir):
        f = codecs.open(work_dir + file_name, mode = 'r')
        headline = f.readline() # 标题行，写入每一个文件中
        length = int(headline.split()[0])
    result = vob1.evaluate_one_base(work_dir, length)
    prediction = pd.DataFrame(columns = name, data = result) # 输出文件，若只想看评测参数请注释掉本行和下一行
    prediction.to_csv(work_dir + "prediction.csv")
    result_list = []
    for i in result:
        temp_result = i[0], i[1], i[2]
        result_list.append(temp_result)
    nums, bases, matches = ipd.Get_Batch_Data(work_dir)
    final_pre = fp.Nus_p(result_list, bases[0])
    pre_match = fp.change_to_match(final_pre)
    match = []
    for mat in matches[0]:
        match.append(int(mat))
    TP, FN, FP, R, P, F1 = fp.estimate(pre_match, match)
    print('查全率：' + str(R))
    print('查准率：' + str(P))
    print('综合衡量：' + str(F1))
    print("将在浏览器中显示图形结果，请等待")
    fp.open_in_webbrowser(final_pre, bases[0]) # 开启默认浏览器
