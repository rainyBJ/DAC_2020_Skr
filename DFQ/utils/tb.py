import os
import numpy as np

def dec2bin(num, bp):
    bp = bp-1
    l = []
    l_tmp = []

    if num < 0:
        sign = '1'
        num = -num
    else:
        sign = "0"
    l.append(sign)

    while num > 0:
        num, remainder = divmod(num, 2)
        l_tmp.append(str(remainder))
    l_tmp.reverse()

    if(len(l_tmp) > bp):#对于超过表示范围的数据进行截断
        if(sign == '0'):#有符号数最大为 0111_1111
            for i in range(bp):
                l.append('1')
        else:#有符号数最小为 1000_0000
            for i in range(bp):
                l.append('0')
    else:
        for i in range(bp-len(l_tmp)):
            l.append('0')
        for i in range(len(l_tmp)):
            l.append(l_tmp[i])

    return ''.join(l)

def dec2binary(signed,num,bits):
    if signed:
        if num > 2**(bits-1)-1:
            num = 2**(bits-1)-1
        if num < -2**(bits-1):
            num = -2**(bits-1)

    else:#对无符号数而言 需要clip操纵
        if num > 2**bits-1:
            num = 2**bits-1

    s = np.binary_repr(num,bits)
    return s
"""
用于做一些测试
"""
if __name__ == "__main__":
    # print(dec2binary(1,32,6))
    # b = bytes([59])
    # print(type(b),b)
    # a = bytes([57])
    # print(type(a), a)
    #
    # with open('test.wb','wb') as f:
    #     f.write(b)
    #     f.write(a)
    # f.close()
    # print(1)
    print(np.binary_repr(1960,8))