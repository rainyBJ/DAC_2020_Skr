import os
from utils.quantize import *
import numpy as np

targ_layer = [QuantNConv2d, QuantNLinear]
saList = []  # for scales of activation
swList = []
sbList = []
wList_reorg = []  # weight param
bList_reorg = []
nm = 17 #bit-width for multiplier


def dec2binary(signed,num,bits):
    """
    convert a int/uint number to binary representation.
    notice: for int6 weight, although we clip it to [int6_min, int6_max],
    we use a 8 bit representation.
    """
    if signed:
        if num > 2**(bits-1)-1:
            if num!=32 :#由于是四舍五入的量化方式所以会有量化后的值为32的情况
                print('overflow')
            num = 2**(bits-1)-1
        if num < -2**(bits-1):
            num = -2**(bits-1)
            print('overflow')
    else:#对无符号数而言 需要clip操纵
        if num > 2**bits-1:
            num = 2**bits-1
            print('overflow')
    if bits < 8:#weight
        s = np.binary_repr(num,8)#权重统一用8位表示
    else:
        s = np.binary_repr(num,bits)
    return s


def get_scale(signed, num_bits, max_value, min_value, sList, min_thresh=1e-5, max_thresh=1e-1,extend=False,extend_num=4):  # get Scale factors
    if signed:#有符号量化
        qmax = 2 ** (num_bits - 1) - 1
        if extend:
            qmax = qmax + extend_num
        max_value = abs(max_value)
        min_value = abs(min_value)
        if max_value > min_value:
            scale = max_value / qmax  # scale factor for weight; Sw
        else:
            max_value = min_value
            scale = max_value / (qmax + 1)
        min_value = 0.
        if scale < min_thresh:#加入阈值 限制scale不会太小 这样bias量化的时候就不会越界了
            if scale != 0:
                print(scale)
                print("越界！")
                scale = min_thresh
            else:
                scale = min_thresh
    else:#activation 的量化
        qmin = 0.
        qmax = 2. ** num_bits - 1.
        scale = (max_value - min_value) / (qmax - qmin)
        if scale < min_thresh:
            scale = min_thresh
            print("越界！")
        if scale > max_thresh:
            scale = max_thresh
            print("越界！")#舍弃个别通道中过大的激活值
    sList.append(scale)


def save_scales_activation(graph, activation_bit):  # scale factors for activations of each layer in the network
    for layer_idx in graph:
        layer = graph[layer_idx]
        if type(layer) in targ_layer:
            running_max = getattr(layer, 'quant').running_max
            running_min = getattr(layer, 'quant').running_min
            get_scale(0, activation_bit, float(running_max), float(running_min), saList)


def save_scales_weight(graph, weight_bit, perchannel):  # scale factors for weights of each channel in the network
    index_conv = 0
    for layer_idx in graph:
        SwList_channel = []
        layer = graph[layer_idx]
        if type(layer) in targ_layer:
            if index_conv==7 or index_conv==9 or index_conv==11:
                extend = True
            else:
                extend = False
            weight = layer.weight.clone()
            for i in range(weight.size()[0]):  # 只考虑 output_channel 维度
                list_tmp = []  # scale factor for each channel
                if perchannel:
                    get_scale(1, weight_bit, float(weight[i].max()), float(weight[i].min()), list_tmp)
                else:
                    get_scale(1, weight_bit, float(weight.max()), float(weight.min()), list_tmp)
                SwList_channel.extend(list_tmp)
            # print("Scale Range for layer"+str(layer_idx)+": ")
            # print(min(SwList_channel),max(SwList_channel))
            index_conv = index_conv + 1
            swList.append(SwList_channel)


def save_scales_bias(saList, swList):  # scale factors for bias of each channel in the network
    if len(saList) == len(swList):  # layer
        for i in range(len(saList)):
            tmp_list = []
            for j in range(len(swList[i])):  # channel
                tmp_list.append(saList[i] * swList[i][j])
            sbList.append(tmp_list)
    else:
        print("Wrong!")


def save_Qparameter(graph, weight_bit, bias_bit):# save parameter in reordered form
    index = 0  # targ_layer 在所有 targ_layers 中的 index
    w_3x3_count = 0  # # of parameters for 3x3 d-conv
    w_1x1_count = 0  # # of parameters for 1x1 p-conv
    b_count = 0  # # of parameters for bias
    m_last_layer = []
    multiple_max = 0
    weight_param = {}  # a dict for conv name and their params
    for layer_idx in graph:
        layer = graph[layer_idx]
        if type(layer) in targ_layer:
            weight = layer.weight.clone()
            bias = layer.bias.clone()

            output_channel = weight.size()[0]
            input_channel = weight.size()[1]
            kernel_size = weight.size()[2]

            w_tmp_list = []
            m_tmp_list = []
            b_tmp_list = []

            if input_channel != 1:  # p-conv
                if output_channel % 32 != 0:  # 输入输出通道都需要变为 32 的整数倍
                    o_ = output_channel + 32 - output_channel % 32  # o_, i_: aligned number of channels for o_c, i_c
                else:
                    o_ = output_channel
                if input_channel % 32 != 0:
                    i_ = input_channel + 32 - input_channel % 32
                else:
                    i_ = input_channel
                w_1x1_count = w_1x1_count + i_ * o_  # 记录参数的数量
                weight_param['conv'+str(index)] = [o_,i_,1,1]
                b_count = b_count + o_
                for o in range(o_):
                    if o < output_channel:
                        weight[o].div_(swList[index][o])  # quantize weight and bias
                        bias[o].div_(sbList[index][o])
                        if index < len(saList) - 1:  # 层间重量化因子 m
                            multiple = sbList[index][o] / saList[index + 1]
                            if multiple > multiple_max:
                                multiple_max = multiple
                            m = round(2 ** nm * multiple)
                            m = dec2binary(1, m,bias_bit)  # 如果后续还有 conv 计算需要 requantization
                        else:
                            m = round(2 ** nm * sbList[index][o])
                            m_last_layer.append(m)  # 用来记录最后一层的缩放因子 m，这里不需要考虑下一层的 sa
                            m = dec2binary(1, m, bias_bit)  # 后续没有 conv 计算只需要 dequantization
                        b = round(float(bias[o]))  # round, float is used to change tensor to float number
                        b = dec2binary(1, b, bias_bit)  # 转换成量化比特位长的二进制数（以字符串形式表示）
                        b_tmp_list.append(b)
                        m_tmp_list.append(m)
                        for i in range(i_):
                            if i < input_channel:
                                for k1 in range(kernel_size):
                                    for k2 in range(kernel_size):
                                        w = round(float(weight[o][i][k1][k2]))
                                        w = dec2binary(1, w, weight_bit)
                                        w_tmp_list.append(w)
                            else:  # 对于输入通道的填充操作
                                w = dec2binary(1, 0, 8)
                                w_tmp_list.append(w)
                    else:  # 超出的部分填充为 0
                        b = dec2binary(1, 0, bias_bit)  # 转换成量化比特位长的二进制数（以字符串形式表示）
                        b_tmp_list.append(b)
                        m_tmp_list.append(b)
                        for i in range(i_):
                            w = dec2binary(1, 0, 8)
                            w_tmp_list.append(w)
            else:  # d-conv
                if output_channel % 32 != 0:  # 只需要变输出通道为 32 的整数倍，输入通道数恒为 1
                    o_ = output_channel + 32 - output_channel % 32
                else:
                    o_ = output_channel
                i_ = input_channel
                w_3x3_count = w_3x3_count + o_ * 3 * 3  # 用来记录参数的数量
                weight_param['conv' + str(index)] = [o_, i_, 3, 3]
                b_count = b_count + o_
                for o in range(o_):  # 对于原来有数据的部分记录下来
                    if o < output_channel:
                        weight[o].div_(swList[index][o])  # quant weight and bias
                        bias[o].div_(sbList[index][o])
                        if index < len(saList) - 1:  # 计算做 requantization 的因子
                            multiple = sbList[index][o] / saList[index + 1]
                            if multiple > multiple_max:
                                multiple_max = multiple
                            m = round(2 ** nm * multiple)
                            m = dec2binary(1, m, bias_bit)
                        else:
                            m = dec2binary(1, round(2 ** nm * sbList[index][o]), bias_bit)
                        b = round(float(bias[o]))  # round
                        b = dec2binary(1, b, bias_bit)  # 转换成量化比特位长的二进制数（以字符串形式表示）
                        b_tmp_list.append(b)
                        m_tmp_list.append(m)
                        for i in range(input_channel):
                            for k1 in range(kernel_size):
                                for k2 in range(kernel_size):
                                    w = round(float(weight[o][i][k1][k2]))
                                    w = dec2binary(1, w, weight_bit)
                                    w_tmp_list.append(w)
                    else:  # 超出的部分填充为 0
                        b = dec2binary(1, 0, bias_bit)  # 转换成量化比特位长的二进制数（以字符串形式表示）
                        b_tmp_list.append(b)
                        m_tmp_list.append(b)
                        for i in range(input_channel):
                            for k1 in range(kernel_size):
                                for k2 in range(kernel_size):
                                    w = dec2binary(1, 0, 8)
                                    w_tmp_list.append(w)

            index = index + 1

            w_reorg(w_tmp_list, i_, o_)  # 将数据补为总线位宽256bit
            b_reorg(b_tmp_list, m_tmp_list, o_)

            print('-' * 10)
            print('input & output channel: ')
            print(i_,o_)
            print()
    print("max multiple:")
    print(multiple_max)
    print("biasm of last layer:")
    print("bbox_m = ",end='')
    print(m_last_layer)
    print("number of weight: ")
    print(w_1x1_count + w_3x3_count)
    print("number of biasm: ")
    print(b_count * 2)


def w_reorg(wList, input_channel, output_channel):  # reorg to 256 bit, 在硬件中每次读入总线位宽的数据256bit，由32个通道拼接得到
    if input_channel != 1:  # p-conv 对应读取时地址逻辑得到写入时的顺序
        in_num_group = int(input_channel / 32)
        out_num_group = int(output_channel / 32)
        for m in range(out_num_group):  # 最外层是 m，out_channel 有多少个 32 group
            for n in range(in_num_group):
                for i in range(32):  # i_c index
                    bus_data = ''
                    for j in range(32):  # o_c index 每32个通道的数据合并
                        index = (m * 32 + j) * input_channel + n * 32 + i
                        bus_data = bus_data + wList[index]
                    wList_reorg.append(bus_data)
    else:  # d-conv
        kernel_size = 3
        num_group = int(output_channel / 32)
        for n in range(num_group):
            for k1 in range(kernel_size):
                for k2 in range(kernel_size):
                        bus_data = ''
                        for i in range(32):  # o_c index
                            index = (32 * n + i) * kernel_size ** 2 + k1 * kernel_size + k2
                            bus_data = bus_data + wList[index]
                        wList_reorg.append(bus_data)


def b_reorg(bList, mList, output_channel):  # reorg to 256bit
    n = int(output_channel / 16)
    for i in range(n):  # bias
        bus_data = ''
        for j in range(16):
            index = i * 16 + j
            bus_data = bus_data + bList[index]
        bList_reorg.append(bus_data)
    for i in range(n):  # multiplier for requantization/dequantization
        bus_data = ''
        for j in range(16):
            index = i * 16 + j
            bus_data = bus_data + mList[index]
        bList_reorg.append(bus_data)


def write2binary(file_name, list_reorg,f_bin):  # 每 8 位为一组，生成对应的 byte 数据，写入二进制文件中
    with open(file_name, 'wb') as f:
        for i in range(len(list_reorg)):
            s = list_reorg[i]
            for _ in range(int(len(s) / 8)):
                binary_str = s[:8]
                decimal_int = int(binary_str, 2)
                b = bytes([decimal_int])
                f.write(b)
                f_bin.write(b)
                s = s[8:]
    f.close()

def write2binary_bm(file_name, list_reorg, f_bin):  # 每 8 位为一组，生成对应的 byte 数据，写入二进制文件
    with open(file_name, 'wb') as f:
        for i in range(len(list_reorg)):
            s = list_reorg[i]
            for _ in range(int(len(s) / 16)):
                binary_str = s[:16]
                byte1 = binary_str[8:]  # 先存低位的 byte 再存高位的 byte
                decimal_int1 = int(byte1, 2)
                b = bytes([decimal_int1])
                f.write(b)
                f_bin.write(b)
                byte2 = binary_str[:8]
                decimal_int2 = int(byte2, 2)
                b = bytes([decimal_int2])
                f.write(b)
                f_bin.write(b)
                s = s[16:]
    f.close()


"""
主函数
生成 HLS 所需要的
 .wt 文件（Qweight 二进制数据）8比特无符号数
 .bm 文件（Qbias Qmult 二进制数据）6比特有符号数
"""
def save_scales_Qparam(graph, activation_bit, weight_bit, bias_bit):
    print("***Save Quantization parameters***")
    # print(os.getcwd())
    save_scales_activation(graph, activation_bit)#scale factor for activations
    print("Range for scale_activation: ")
    print(saList)
    print(min(saList),max(saList))
    save_scales_weight(graph, weight_bit,perchannel=0)#scale factor for weights 最后一个参数可以调整是按层量化 还是按通道量化
    print("Range for scale_weight: ")
    print(min(min(x) for x in swList),max(max(x) for x in swList))
    save_scales_bias(saList, swList)#scale factor for bias
    print("Range for scale_bias: ")
    print(min(min(x) for x in sbList),max(max(x) for x in sbList))
    save_Qparameter(graph, weight_bit, bias_bit)#write Qw Qb Qm to wList_reorg & bList_reorg
    # print("Qweight")
    # print(len(wList_reorg), wList_reorg[0])
    # print("Qbiasm")
    # print(len(bList_reorg), bList_reorg[0])
    with open('Quantized_parameter/skr.bin','wb') as f:  # write binary parameters to skr.bin file, weight in skr.wt, bias&multi in skr.bm
        write2binary('Quantized_parameter/skr.wt', wList_reorg, f)
        write2binary_bm('Quantized_parameter/skr.bm', bList_reorg, f)
    f.close