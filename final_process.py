import numpy as np
import webbrowser
import pandas as pd
import numpy as np
import csv


import input_and_process_data as ipd

'''
输入：长度为n的list，每一项是一个（float，float，float）的元组
     对应一条长度为n的rna中每个碱基对应点括号三种情况的概率
     以及文件路径

输出：对应一个rna的，符合实际情况的最优预测序列
     预测正确的碱基个数
     rna长度

调用方法：
         prediction = csv_to_prediction(PATH)
         CSV为在test.py中保存结果的csv文件
         PATH为配对概率文件CSV的准确路径名
         
         final_pre = Nus_p(prediction, path)
         final_pre为最终预测结果(点括号构成的序列)

         pre_match = change_to_match(final_pre)
         pre_match为由点括号序列转换的配对序列

         TP, FP, FN, Recall, Precision, F1 = estimate(pre_match, path)
         TP:正确预测的碱基对个数
         FN:真实结构中存在但没有预测出来的碱基对个数
         FP:真实结构不存在却被错误预测的碱基对个数
         R:所有预测到的碱基对中正确的百分比（查准率）
         P:真实结构所有碱基对中被预测出来的百分比（查全率）
         F1:衡量查全率和查准率的参数
         R = TP/(TP+FP)
         P = TP/(TP+FN)
         F1 = 2*P*R/(P+R)
         以上参数的值可以针对单个rna，也可以在循环中累加最后再计算即为针对所有rna

         open_in_webbrowser(final_pre, bases)
         调用ViennaRNA/forna提供的API在浏览器中显示图形化预测结果
'''


# 判断碱基之间是否两两配对
def is_paired(x, y):
    if x == 'A' and y == 'U':
        return True
    elif x == 'G' and y == 'C':
        return True
    elif x == "G" and y == 'U':
        return True
    elif x == 'U' and y == 'A':
        return True
    elif x == 'C' and y == 'G':
        return True
    elif x == "U" and y == 'G':
        return True
    else:
        return False


# 记录由碱基对配对关系决定的，参与计算时的取值,bases是碱基序列
def count_paired(prediction, bases):
    l = len(prediction)
    r = np.zeros((l, l))
    for i in range(l):
        for j in range(l):
            if is_paired(bases[i], bases[j]):
                r[i, j] = prediction[i][0] + prediction[j][2]
            else:
                r[i, j] = prediction[i][1] + prediction[j][1]
    return r


# 借鉴Nussinov算法，迭代计算从i到j的序列最大概率和，path是文件路径
# 正式使用：将bases换成path，函数第一行的井号删掉即可
def Nus_p(prediction, bases):
    # nums, bases, matches = ipd.Get_Batch_Data(path)
    r = count_paired(prediction, bases)
    l = len(prediction)
    final_pre = []
    n = np.zeros((l, l))
    result = [[''] * l for i in range(l)]
    for k in range(l):
        i = 0
        while i + k < l:
            # 对n(i,i+k)操作
            j = i + k
            # 对n（i,j)操作
            max1 = 0
            max3 = 0
            if i + 1 < l and j - 1 >= 0:
                max1 = n[i + 1][j] + prediction[i][1]
                max3 = n[i + 1][j - 1] + r[i][j]  # 只有这种情况对应配对
            max2 = n[i][j - 1] + prediction[j][1]
            max4 = 0
            for x in range(i + 1, j - 1, 1):
                if max4 < n[i][x] + n[x + 1][j]:
                    max4 = n[i][x] + n[x + 1][j]  # 可能为分割成两个子序列
                    point = x
            m = max(max1, max2, max3, max4)
            n[i][j] = m
            if m == max1:
                result[i][j] = '.' + result[i + 1][j]
            elif m == max2:
                result[i][j] = result[i][j - 1] + '.'
            elif m == max3:
                if len(result[i + 1][j - 1]) == j - i - 1:
                    if is_paired(bases[i], bases[j]):
                        result[i][j] = '(' + result[i + 1][j - 1] + ')'  # 可以保证产生的预测序列左右括号必然一一对应
                    else:
                        result[i][j] = '.' + result[i + 1][j - 1] + '.'
                else:
                    result[i][j] = '.'
            elif m == max4:
                result[i][j] = result[i][point] + result[point + 1][j]
            i = i + 1
    str = result[0][l - 1]
    for c in str:
        final_pre.append(c)
    return final_pre


# 将上一步获得的点括号序列转换成配对序列便于后续计算
def change_to_match(final_pre):
    stack = []  # 用列表代替堆栈（代替链表？）
    pre_match = [0] * len(final_pre)  # 返回的配对序列
    for i in range(len(final_pre)):
        if final_pre[i] == '(':
            stack.append(i)
        elif final_pre[i] == '.':
            pre_match[i] = 0  # 0表示没有配对
        else:
            pre_match[i] = stack[len(stack) - 1] + 1
            pre_match[stack[len(stack) - 1]] = i + 1
            stack.pop()
    return pre_match


# 后续计算 参数含义见文件头
# 正式使用：将matches换成path，函数第一行的井号删掉即可
def estimate(pre_match, matches):
    # nums, bases, matches = ipd.Get_Batch_Data(path)
    TP = 0
    FN = 0
    FP = 0
    F1 = 0
    for i in range(len(pre_match)):
        if pre_match[i] == matches[i] and pre_match[i] >= i:  # 最后这个条件防止重复计算
            TP = TP + 1
        # 如果预测3-23，但实际上3无配对，则仅FP加一
        # 如果预测3无配对，但实际上3-23，则仅FN加一
        # 如果预测3-23，但实际上3-22，则FN和FP均加一
        if pre_match[i] != matches[i] and matches[i] != 0 and matches[i] >= i:
            FN = FN + 1
        if pre_match[i] != matches[i] and pre_match[i] != 0 and pre_match[i] >= i:
            FP = FP + 1
    R = TP / (TP + FP)  # 查全率
    P = TP / (TP + FN)  # 查准率
    if R != 0 or P != 0:
        F1 = 2 * P * R / (P + R)
    return TP, FN, FP, R, P, F1

#将CSV转换成prediction的函数，PATH为CSV的准确路径名如:C:\FileRecv\5s_Acetobacter-sp.-1\prediction.csv
def csv_to_prediction(PATH):
    data = pd.read_csv(PATH, sep=',',
        header=0, index_col=0)
    array_data = np.array(data)
    list_data = array_data.tolist()
    prediction = []
    for list in list_data:
        prediction.append(tuple(list))
    return prediction

# 利用ViennaRNA/forna提供的API在浏览器中显示图形化预测结果
def open_in_webbrowser(final_pre, bases):
    seq = "".join(bases)
    stu = "".join(final_pre)
    url = 'http://nibiru.tbi.univie.ac.at/forna/forna.html?id=url/name&sequence=' + seq + '&structure=' + stu
    #print(url)
    webbrowser.open(url)


# 以下为测试用真实数据
# path = ''

'''
    对CT文件和运行得出的csv文件进行测试
    
prediction=csv_to_prediction(r'C:\Users\miemiemie\Documents\Tencent Files\1206198069\FileRecv\5s_Acetobacter-sp.-1\prediction.csv')
nums, bases, matches = ipd.Get_Batch_Data(r'C:\Users\miemiemie\Documents\Tencent Files\1206198069\FileRecv\代码-CDPfold\data\5sRNA\5sRNA_ct\\')
base = bases[0]
matche = []
for mat in matches[0]:
    matche.append(int(mat))
print(len(base),len(matche))
print(matche)
print(base)
print(prediction)

'''


'''
base = ['C', 'C', 'U', 'C', 'C', 'C',
         'U', 'U', 'G', 'G', 'G', 'G',
         'C', 'A', 'G', 'G', 'G', 'G',
         'A', 'G', 'G', 'C', 'U', 'G']
prediction = [(1, 0, 0), (1, 0, 0), (1, 0, 0), (1, 0, 0), (1, 0, 0), (1, 0, 0),
              (1, 0, 0), (1, 0, 0), (1, 0, 0), (1, 0, 0), (0, 1, 0), (0, 1, 0),
              (0, 1, 0), (0, 1, 0), (0, 1, 0), (0, 1, 0), (0, 1, 0), (0, 0, 1),
              (0, 0, 1), (0, 0, 1), (0, 0, 1), (0, 0, 1), (0, 0, 1), (0, 0, 1),
              (0, 0, 1), (0, 0, 1), (0, 0, 1), (0, 1, 0), (0, 1, 0), (0, 1, 0)]
matche = [27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 0, 0, 0, 0, 0, 0, 0, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 0, 0]
final_pre = Nus_p(prediction, base)  # 正式运行时bases应该为path
print(final_pre)
pre_match = change_to_match(final_pre)
print(pre_match)
print(len(final_pre),len(pre_match))
TP, FN, FP, R, P, F1 = estimate(pre_match, matche)  # 正式运行时matches应该为path
print('查全率：' + str(R))
print('查准率：' + str(P))
print('综合衡量：' + str(F1))
print("将在浏览器中显示图形结果，请等待")
open_in_webbrowser(final_pre, base)

'''
