# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 19:36:24 2019

@author: 雷浩洁
"""

import numpy as np
import pandas as pd
import tensorflow as tf
import codecs
import os



def Get_Batch_Data(filepath):
    """
    批量读取一个文件夹里所有文件的数据
    参数为文件夹的路径
    返回三个大列表，分别为
    nums:保存所有的RNA序列信息，列表的每一个元素是一个RNA的序列信息
    bases:保存所有的RNA碱基信息,列表的每一个元素是一个RNA的碱基信息
    matchs:保存所有的RNA配对信息,列表的每一个元素是一个RNA的对应碱基的配对信息
    """
    pathDir=os.listdir(filepath)
    nums=[]#保存第一列序列信息，格式为int,每一个元素都是一个列表，保存一条序列的信息
    bases=[]#保存第二列碱基序列信息，为ACGU字符串,每一个元素都是一个列表，保存一条序列的信息
    matchs=[]#保存第五列配对信息，格式为int,每一个元素都是一个列表，保存一条序列的信息
    for file_name in pathDir:
        f=codecs.open(filepath+str(file_name),mode='r')#打开一个文件
        
        num=[]
        base=[]
        match=[]
        
        line=f.readline()#读入一行
        while line:
            a=line.split()
            
            #切片读取第1,2,5列的信息
            num.extend(a[0:1])
            base.extend(a[1:2])
            match.extend(a[4:5])
            
            #读取下一行
            line=f.readline()
        
        #去除无用数据
        num.remove(num[0])
        base.remove(base[0])
        match.remove(match[0])
        
        #添加到大列表中
        nums.append(num)
        bases.append(base)
        matchs.append(match)
        
        f.close()
    return nums,bases,matchs


PATH="All_Clearn_Data/"#想要读取的文件的文件名
nums,bases,matchs=Get_Batch_Data(PATH)#调用方式


"""
#测试用
print(nums[0])
print(bases[0])
print(matchs[0])
print("...")
print("...")
print("...")
print(nums[1000])
print(bases[1000])
print(matchs[1000])
"""
        