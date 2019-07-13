import numpy as np
from PIL import Image
# 以下两个包仅用于主函数测试用
import input_and_process_data as ipd
import coding_matrix as cm

'''
输入：m个n*n矩阵形成的list
输出：m*n个大小为num*length的矩阵形成的list
调用方法：result=slide_window(matrix, num)
         调用后result为长度为m*n的list 其中每一项都是一个行数为row列数不定的矩阵(row暂时用11或19)
         result=same_size(result, num ,length)
         调用后result为长度为m*n的list 其中每一项都是一个row*column的矩阵(column暂时用120或128)
'''
def slide_window(matrix, row):
    result = []
    for x in range(len(matrix)):
        more = np.zeros((len(matrix[x]) + row, len(matrix[x])))
        for i in range(len(matrix[x])):
            more[i + int(0.5*(row - 1))] = matrix[x][i]  # 循环结束后more矩阵就是matrix矩阵底下加了10行0
            for k in range(len(matrix[x])):
                result.append(more[k:k + row])  # 取11行为一项
    return result


def same_size(matrix, column):
    for x in range(len(matrix)):
        image = Image.fromarray(matrix[x])
        image = image.resize((column, len(matrix[x])))
        matrix[x] = (np.asarray(image))
    return matrix

np.set_printoptions(threshold=np.inf, formatter={'float': '{:.1f}'.format})
bases = ['A','G','C','C','G','U']
result = cm.coding_matrix(bases)
result = slide_window(result, 19)
result = same_size(result, 128)
fina = np.array(result)
fina = fina.reshape(len(fina), 19, 128, 1)
print(fina)
