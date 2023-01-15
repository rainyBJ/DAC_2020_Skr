import sys
sys.path.append('c:\\Users\\44724\\gitrepos\\cpipc')
from DFQ.utils.int_parameters import *
from torch.utils.data import Dataset
import torchvision.transforms as transforms

from utils.relation import create_relation
from dfq import cross_layer_equalization, bias_absorption, bias_correction, clip_weight
from utils.layer_transform import switch_layers, replace_op, set_quant_minmax, merge_batchnorm, quantize_targ_layer
from PyTransformer.transformers.torchTransformer import TorchTransformer
from utils.quantize import QuantConv2d, QuantLinear, QuantNConv2d, QuantNLinear, QuantMeasure, QConv2d, QLinear, set_layer_bits
from improve_dfq import update_quant_range, set_update_stat
from modeling.detection.skynet import *

from PIL import Image
import argparse
import numpy as np
import time
import os
print(f'current dir:{os.getcwd()}')
import h5py
from torch.autograd import Variable
import datasetTest

parser = argparse.ArgumentParser(description="SkyNet Evaluation on DAC dataset.")
parser.add_argument("--quantize", action='store_false')
parser.add_argument("--equalize", action='store_true')
parser.add_argument("--correction", action='store_true')
parser.add_argument("--absorption", action='store_true')
parser.add_argument("--distill_range", action='store_true')
parser.add_argument("--log", action='store_true')
parser.add_argument("--relu", action='store_false')
parser.add_argument("--clip_weight", action='store_true')
parser.add_argument("--trainable", action='store_true')
parser.add_argument("--equal_range", type=float, default=1e8)
parser.add_argument("--bits_weight", type=int, default=6)
parser.add_argument("--bits_activation", type=int, default=8)
parser.add_argument("--bits_bias", type=int, default=16)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument('--workers', type=int, default=10, metavar='N')
args = parser.parse_args()
DEVICE = torch.device("cuda:0")

def load_net(fname, net):
    print(f'file name:{fname}')
    with h5py.File(fname, 'r') as h5f:
        for k, v in net.state_dict().items():
            if 'num_batches' in k:
                continue
            if 'model_p3.1.bias' in k:
                param = torch.from_numpy(np.zeros((10))).float()#转换成float32
                v.copy_(param)
                continue
            param = torch.from_numpy(np.asarray(h5f[k]))
            v.copy_(param)

def find_min_max(net):#conv layers in net
    min_weight = 0
    max_weight = 0
    min_bias = 0
    max_bias = 0
    for k,v in net.state_dict().items():
        min_layer = torch.min(v)
        max_layer = torch.max(v)
        #skynet 中所有的 dw block 中 conv 层所在位置为 0.weight 3.weight； bias 层所在位置为 0.bias 3.bias
        if '0.weight' in k or '3.weight' in k or 'model_p3.1.weight' in k:
            # print('min/max weight of this layer: ', end='')
            # print(min_layer, max_layer)
            if min_weight > min_layer:
                min_weight = min_layer
            if max_weight < max_layer:
                max_weight = max_layer
        if '0.bias' in k or '3.bias' in k or 'model_p3.1.bias' in k:
            # print('min/max bias of this layer: ', end='')
            # print(min_layer, max_layer)
            if min_bias > min_layer:
                min_bias = min_layer
            if max_bias < max_layer:
                max_bias = max_layer
    print('min/max weight of all layers: ',end='')
    print(min_weight,max_weight)
    print('min/max bias of all layers: ', end='')
    print(min_bias, max_bias)

def convert2cpu(gpu_matrix):
    return torch.FloatTensor(gpu_matrix.size()).copy_(gpu_matrix)


def test(net):

    cur_model = net
    anchors = cur_model.anchors
    num_anchors = cur_model.num_anchors
    anchor_step = int(len(anchors) / num_anchors)
    total = 0.0
    proposals = 0.0

    for batch_idx, (data, target) in enumerate(test_loader):
        with torch.no_grad():
            data = data.cuda()
            data = Variable(data)
            output = net(data).data
            batch = output.size(0)
            h = output.size(2)
            w = output.size(3)
            output = output.view(batch * num_anchors, 5, h * w).transpose(0, 1).contiguous().view(5,
                                                                                                  batch * num_anchors * h * w)
            grid_x = torch.linspace(0, w - 1, w).repeat(h, 1).repeat(batch * num_anchors, 1, 1).view(
                batch * num_anchors * h * w).cuda()
            grid_y = torch.linspace(0, h - 1, h).repeat(w, 1).t().repeat(batch * num_anchors, 1, 1).view(
                batch * num_anchors * h * w).cuda()
            xs = torch.sigmoid(output[0]) + grid_x
            ys = torch.sigmoid(output[1]) + grid_y

            anchor_w = torch.Tensor(anchors).view(num_anchors, anchor_step).index_select(1, torch.LongTensor([0]))
            anchor_h = torch.Tensor(anchors).view(num_anchors, anchor_step).index_select(1, torch.LongTensor([1]))
            anchor_w = anchor_w.repeat(batch, 1).repeat(1, 1, h * w).view(batch * num_anchors * h * w).cuda()
            anchor_h = anchor_h.repeat(batch, 1).repeat(1, 1, h * w).view(batch * num_anchors * h * w).cuda()
            ws = torch.exp(output[2]) * anchor_w
            hs = torch.exp(output[3]) * anchor_h
            det_confs = torch.sigmoid(output[4])
            sz_hw = h * w
            sz_hwa = sz_hw * num_anchors
            det_confs = convert2cpu(det_confs)
            xs = convert2cpu(xs)
            ys = convert2cpu(ys)
            ws = convert2cpu(ws)
            hs = convert2cpu(hs)

            for b in range(batch):
                det_confs_inb = det_confs[b * sz_hwa:(b + 1) * sz_hwa].numpy()
                xs_inb = xs[b * sz_hwa:(b + 1) * sz_hwa].numpy()
                ys_inb = ys[b * sz_hwa:(b + 1) * sz_hwa].numpy()
                ws_inb = ws[b * sz_hwa:(b + 1) * sz_hwa].numpy()
                hs_inb = hs[b * sz_hwa:(b + 1) * sz_hwa].numpy()
                ind = np.argmax(det_confs_inb)

                bcx = xs_inb[ind]
                bcy = ys_inb[ind]
                bw = ws_inb[ind]
                bh = hs_inb[ind]
                box = [bcx / w, bcy / h, bw / w, bh / h]

                iou = bbox_iou(box, target[b][1:5], x1y1x2y2=False)
                proposals = proposals + iou
                total = total + 1

    avg_ious = proposals / total
    print("iou: %f" % avg_ious)

    return avg_ious

def bbox_iou(box1, box2, x1y1x2y2=True):
    if x1y1x2y2:
        mx = min(box1[0], box2[0])
        Mx = max(box1[2], box2[2])
        my = min(box1[1], box2[1])
        My = max(box1[3], box2[3])
        w1 = box1[2] - box1[0]
        h1 = box1[3] - box1[1]
        w2 = box2[2] - box2[0]
        h2 = box2[3] - box2[1]
    else:
        mx = min(box1[0]-box1[2]/2.0, box2[0]-box2[2]/2.0)
        Mx = max(box1[0]+box1[2]/2.0, box2[0]+box2[2]/2.0)
        my = min(box1[1]-box1[3]/2.0, box2[1]-box2[3]/2.0)
        My = max(box1[1]+box1[3]/2.0, box2[1]+box2[3]/2.0)
        w1 = box1[2]
        h1 = box1[3]
        w2 = box2[2]
        h2 = box2[3]
    uw = Mx - mx
    uh = My - my
    cw = w1 + w2 - uw
    ch = h1 + h2 - uh

    if cw <= 0 or ch <= 0:
        return 0.0

    area1 = w1 * h1
    area2 = w2 * h2
    carea = cw * ch
    uarea = area1 + area2 - carea
    return carea/uarea

if __name__ == '__main__':
    # print(os.getcwd())
    assert args.relu or args.relu == args.equalize, 'must replace relu6 to relu while equalization'
    assert args.equalize or args.absorption == args.equalize, 'must use absorption with equalize'
    print(args)

    # load_state_dict 载入模型
    net = SkyNet()
    net.eval()#载入模型后必须以 eval 状态进行 DFQ 的相关操作！！！
    # load_net('./modeling/detection/DAC_SkyNet.weights',net)#从h5f中读入权重数据
    # load_net('./modeling/detection/dac.weights', net)
    # load_net('./modeling/detection/01_SkyNet.weights', net)
    # load_net('./modeling/detection/01_sar_SkyNet.weights', net)#没有bias 从dac.weight训练得到 0.7022
    # load_net('./modeling/detection/SAR_0.7011.weights', net)#有bias 自己训练得到
    # load_net('./modeling/detection/01_sar_QAT_SkyNet.weights', net)
    # load_net('./modeling/detection/01_DAC_QAT_SkyNet.weights', net)
    model_dict_path = os.path.join(os.getcwd(), 'modeling\\detection\\01_DAC_QAT_SkyNet.weights')  # 73.64; without last layer's bias; without input normalization
    # model_dict_path = os.path.join(os.getcwd(), 'modeling\\detection\\dac.weights')  # 71.44; without last layer's bias; with input normalization
    # model_dict_path = os.path.join(os.getcwd(), 'modeling\\detection\\DAC_SkyNet.weights')  # 62.33 with bias
    load_net(model_dict_path, net)
    # state_dict = torch.load(model_dict_path)
    # net.load_state_dict(state_dict)
    # convert to gpu
    if torch.cuda.is_available():
        net.cuda()
    # for np, p in net.named_parameters():
    #     print(np)

    # load_net(model_dict_path, net)
    init_width = net.width
    init_height = net.height
    testlist = 'DAC_test.txt'  # DAC_test images information
    # testlist = 'SAR_test.txt'#SAR_test images information
    # test_loader = torch.utils.data.DataLoader(
    #     datasetTest.listDataset(testlist, shape=(init_width, init_height),
    #                             shuffle=False,
    #                             transform=transforms.Compose([
    #                                 transforms.ToTensor(),transforms.Normalize(mean = [ 0.5, 0.5, 0.5 ],std = [ 0.25, 0.25, 0.25 ]),
    #                             ]), train=False),
    #     batch_size=args.batch_size, shuffle=True, num_workers=10, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(
        datasetTest.listDataset(testlist, shape=(init_width, init_height),
                                shuffle=False,
                                transform=transforms.Compose([
                                    transforms.ToTensor(),
                                ]), train=False),
        batch_size=args.batch_size, shuffle=True, num_workers=10, pin_memory=True)

    # test the net
    test(net)


    print("初始参数范围：")
    find_min_max(net)  # 找出权重和激活值中最大和最小值

    # DFQ
    transformer = TorchTransformer()
    module_dict = {}
    if args.quantize:
        if args.distill_range:
            module_dict[1] = [(torch.nn.Conv2d, QConv2d), (torch.nn.Linear, QLinear)]
        elif args.trainable:
            module_dict[1] = [(torch.nn.Conv2d, QuantConv2d), (torch.nn.Linear, QuantLinear)]
        else:
            module_dict[1] = [(torch.nn.Conv2d, QuantNConv2d), (torch.nn.Linear, QuantNLinear)]  # 把卷积层和全连接层换成对应的量化模式
    
    if args.relu:
        module_dict[0] = [(torch.nn.ReLU6, torch.nn.ReLU)]  # 把ReLU6换成ReLU
    
    data = torch.ones((4, 3, 160, 320))
    if torch.cuda.is_available():
        data = data.cuda()
    net, transformer = switch_layers(net, transformer, data, module_dict, ignore_layer=[QuantMeasure], quant_op=args.quantize)  # 把网络中原来的层换成对应的量化形式
    graph = transformer.log.getGraph()  # id 对应 module
    bottoms = transformer.log.getBottoms()  # id 对应 id；每一层的上一层是什么
    output_shape = transformer.log.getOutShapes()  # 每一层输出的形状

    if args.quantize:
        if args.distill_range:
            targ_layer = [QConv2d, QLinear]
        elif args.trainable:
            targ_layer = [QuantConv2d, QuantLinear]
        else:
            targ_layer = [QuantNConv2d, QuantNLinear]
    else:
        targ_layer = [torch.nn.Conv2d, torch.nn.Linear]

    if args.quantize:
        set_layer_bits(graph, args.bits_weight, args.bits_activation, args.bits_bias, targ_layer)#为 targ_layer 中的 activation 设置比特数，默认是 a8w8b16

    net = merge_batchnorm(net, graph, bottoms, targ_layer)#将BN层融合conv层中（改变了conv层的weight和bias），并将BN层原来的数据记录在fake_weight和fake_bias中
    print("融合BN层后参数范围：")
    find_min_max(net)

    # create relations
    if args.equalize or args.distill_range:
        res = create_relation(graph, bottoms, targ_layer, delete_single=False)#默认把网络中只有两个层的连接关系删掉
        if args.equalize:
            # 是否要用箱状图展示该过程权重的变化 定义收敛条件（*平均变化量；平均变化量的变化量） 为缩放因子定最小最大值防止overflow 输出箱状图到weight_distribution文件夹中 这个地方很关键不想让 bias 范围扩太大
            cross_layer_equalization(graph, res, targ_layer, visualize_state=False, converge_thres=1e-6, converge_count=3, s_range=(1/args.equal_range, args.equal_range))
            print("权重均衡后参数范围：")
            find_min_max(net)

    # if args.distill:
    #     set_scale(res, graph, bottoms, targ_layer)

    if args.absorption:
        if bias_absorption(graph, res, bottoms, 3):
            print("高偏置吸收后参数范围：")
            find_min_max(net)

    if args.clip_weight:
        clip_weight(graph, range_clip=[-15, 15], targ_type=targ_layer)

    if args.correction:
        bias_correction(graph, bottoms, targ_layer, bits_weight=args.bits_weight)
        print("量化偏移修正吸收后参数范围：")
        find_min_max(net)


    if args.quantize:
        if args.distill_range:
            set_update_stat(net, [QuantMeasure], True)
            # net = update_quant_range(net.cuda(), data_distill, graph, bottoms, is_detection=True)
            set_update_stat(net, [QuantMeasure], False)
        else:
            set_quant_minmax(graph, bottoms, is_detection=True,N=3)  # 为QuantMeasrue方法设置激活值的阈值 [Beta-6Gamma, Beta+6Gamma] 被激活函数截断；输入的范围直接给出，为[0,1]
            save_scales_Qparam(graph,args.bits_activation,args.bits_weight,args.bits_bias)

        if not args.trainable and not args.distill_range:
            graph = quantize_targ_layer(graph, args.bits_weight, args.bits_bias, targ_layer)  # 以权重和偏移的最大最小值为范围进行对称量化

        torch.cuda.empty_cache()  # 显存释放 把没用的变量所占用的显卡内存释放掉

    # 用DFQ后的模型做验证
    if args.quantize:
        replace_op()  # 把 Tensor_op 如 add 和 concate 等换成量化的形式
    print("Start Inference-int8")

    net = net.to(DEVICE)
    test(net)