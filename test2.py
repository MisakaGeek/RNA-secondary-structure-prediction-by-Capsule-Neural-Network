# -*- coding: utf-8 -*-
"""
Created on Sun Jul 14 22:21:17 2019

@author: 孙博文
测试集
第二版：为每个碱基建立新文件夹和专属ct文件，有大量文件操作，奇慢无比，但是准确率意外地高于第三版
"""
import validation_one_base2 as vob2
import input_and_process_data as ipd
import final_process as fp
import pandas as pd
import os
import codecs

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
    work_dir = test_data_path + ct + '/'
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
        each_file.write(' 1 2 ') # 第三列和第四列，占位用
        each_file.write(matches[0][int(index)-1])
        each_file.close()
        temp_result = vob2.evaluate_one_base(work_dir + index + "/")
        result = temp_result[0][0], temp_result[0][1], temp_result[0][2] # 形成a,b,c三元tuple
        result_list.append(result)
        os.remove(work_dir + index + '/' + index + '.ct')
        os.rmdir(work_dir + index)
    prediction = pd.DataFrame(columns = name, data = result_list) # 输出文件，若只想看评测参数请注释掉本行和下一行
    prediction.to_csv(work_dir + "prediction.csv")
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
