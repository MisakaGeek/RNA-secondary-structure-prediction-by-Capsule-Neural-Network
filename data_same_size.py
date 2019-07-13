import numpy as np
from PIL import Image

'''
输入：m个平均长度为n的方阵形成的list

输出：m*n个大小为row*column的矩阵形成的list

调用方法：result=slide_window(matrix, row)
         调用后result为长度为m*n的list 其中每一项都是一个行数为row列数不定的矩阵(row暂时用11或19)
         
         result=same_size(matrix, column)
         调用后result为长度为m*n的list 其中每一项都是一个row*column的矩阵(column暂时用120或128)
         
         通常连用
'''


# 滑窗算法 统一行数
def slide_window(matrix, row):
    result = []
    for x in range(len(matrix)):
        more = np.zeros((len(matrix[x]) + row, len(matrix[x])))
        for i in range(len(matrix[x])):
            more[i + int(0.5 * (row - 1))] = matrix[x][i]  # 循环结束后more矩阵就是matrix矩阵上下加9行0
        for k in range(len(matrix[x])):
            result.append(more[k:k + row])  # 取19行为一项，每个碱基对应矩阵自己在中间
    return result


# 归一化 统一列数
def same_size(matrix, column):
    for x in range(len(matrix)):
        image = Image.fromarray(matrix[x])  # 将矩阵转化为图像
        image = image.resize((column, len(matrix[x])))  # 用处理图像的resize函数缩放，双括号血泪教训
        matrix[x] = (np.asarray(image))  # 将图像转化回矩阵 即可得到大小一致的矩阵
    return matrix
    
    
