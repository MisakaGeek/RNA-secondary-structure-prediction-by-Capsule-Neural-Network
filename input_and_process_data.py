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
    matches=[]#保存第五列配对信息，格式为int,每一个元素都是一个列表，保存一条序列的信息
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
       # match.remove(match[0])#找到BUG，错误地删除了第五列的第一个元素
        
        #添加到大列表中
        nums.append(num)
        bases.append(base)
        matches.append(match)
        
        f.close()
    return nums,bases,matches






def Change_to_String(nums,bases,matches):
    """
    将得到的RNA序列信息转化为点括号表示法
    输入为三条列表
    nums:保存所有的RNA序列信息，列表的每一个元素是一个RNA的序列信息
    bases:保存所有的RNA碱基信息,列表的每一个元素是一个RNA的碱基信息
    matchs:保存所有的RNA配对信息,列表的每一个元素是一个RNA的对应碱基的配对信息
    
    输出为两条列表，分别为：
    bases:保存所有的RNA碱基信息，列表的每一个元素是一个RNA的碱基信息
    stus:保存所有的点括号序列信息，列表的每一个元素stu是一个RNA的点括号序列信息
    """
    stus=[]
    for rna in range(len(nums)):#rna为索引，代表了每一条RNA的索引值
        num=nums[rna]
        base=bases[rna]
        match=matches[rna]
        stu=[]
        for i in range(len(base)):
            if match[i]=='0':#当前碱基没有配对
                stu.append('.')
            else:
                if int(num[i])>int(match[i]):#配对的后一个碱基，用‘）’表示
                    stu.append(')')
                else:              #配对的前一个碱基，用‘（’表示                    
                    stu.append('(')
        stus.append(stu)
        
    return bases,stus


#测试

