from __future__ import print_function
import sys
if len(sys.argv) != 4:
    print('Usage:')
    print('python train.py model datacfg weightfile')
    exit()
import time
import torch
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import dataset
from utils import *
from models import *
import numpy as np

import os
print(os.getcwd())

# Training settings
modeltype = sys.argv[1]+'()'
trainlist       = sys.argv[2]
weightfile    = sys.argv[3]
# testlist = 'SAR_test.txt'
testlist = 'dji_test.txt'
backupdir = 'backup'
gpus = '0'
ngpus = len(gpus.split(','))
num_workers = 0
batch_size    = 32
learning_rate = 1e-5
momentum      = 0.9
decay         = 1e-5
# decay         = 5e-4
# steps         = [-1,100,20000,30000]
steps         = [8000,10000,12000,13000,14000]
scales        = [.1,.1,.1,.1,.1]
# scales        = [.1,.1,.1,.1]#e-3 e-4 e-5 e-6
#Train parameters
max_epochs = 40
use_cuda      = True
seed          = int(10)
eps           = 1e-5
save_interval = 1  # epoches
# Test parameters
best_iou = 0
torch.manual_seed(seed)
if use_cuda:
    os.environ['CUDA_VISIBLE_DEVICES'] = gpus
    torch.cuda.manual_seed(seed)
model       = eval(modeltype)
region_loss = model.loss
load_net(weightfile,model)
region_loss.seen  = model.seen
processed_batches = 0

init_width        = model.width
init_height       = model.height
init_epoch=0
kwargs = {'num_workers': num_workers, 'pin_memory': True} if use_cuda else {}
test_loader = torch.utils.data.DataLoader(
    dataset.listDataset(testlist, shape=(init_width, init_height),
                   shuffle=False,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                   ]), train=False),
    batch_size=batch_size, shuffle=True, **kwargs)
if use_cuda:
    if ngpus > 1:
        model = torch.nn.DataParallel(model).cuda()
    else:
        model = model.cuda()
params_dict = dict(model.named_parameters())
params = []
for key, value in params_dict.items():
    if key.find('.bn') >= 0 or key.find('.bias') >= 0:
        params += [{'params': [value], 'weight_decay': 0.0}]
    else:
        params += [{'params': [value], 'weight_decay': decay*batch_size}]
        
optimizer = optim.SGD(model.parameters(), lr=learning_rate/batch_size, momentum=momentum, dampening=0, weight_decay=decay*batch_size)

def adjust_learning_rate(optimizer, batch):
    lr = learning_rate
    for i in range(len(steps)):
        scale = scales[i] if i < len(scales) else 1
        if batch >= steps[i]:
            lr = lr * scale  # 0:10^-4 100:10^-3 20k:10^-4 30k:10^-5
            if batch == steps[i]:
                break
        else:
            break
    for param_group in optimizer.param_groups:#batch_size越大？learning_rate越小？
        # param_group['lr'] = lr/batch_size
        param_group['lr'] = lr
    return lr

def train(epoch):
    global processed_batches
    t0 = time.time()
    if ngpus > 1:
        cur_model = model.module
    else:
        cur_model = model
    train_loader = torch.utils.data.DataLoader(
        dataset.listDataset(trainlist, shape=(init_width, init_height),
                            shuffle=True,
                            transform=transforms.Compose([
                                transforms.ToTensor(),
                            ]),
                            train=True,
                            seen=cur_model.seen,
                            batch_size=batch_size,
                            num_workers=num_workers),
        batch_size=batch_size, shuffle=False, **kwargs)
    lr = adjust_learning_rate(optimizer, processed_batches)
    logging('epoch %d, processed %d samples, lr %f, processed_batches: %d' % (epoch, epoch * len(train_loader.dataset), lr, processed_batches))
    model.train()
    
    for batch_idx, (data, target) in enumerate(train_loader):
        lr = adjust_learning_rate(optimizer, processed_batches)
        processed_batches = processed_batches + 1
        if use_cuda:
            data = data.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()#清空梯度
        output = model(data)
        region_loss.seen = region_loss.seen + data.data.size(0)#每次加上一个 batch 的数据 模型见过的数据
        loss = region_loss(output, target)
        loss.backward()
        optimizer.step()
        
    t1 = time.time()
    logging('training with %f samples/s' % (len(train_loader.dataset)/(t1-t0)))
    if (epoch+1) % save_interval == 0:
        model_save_dir = f'{backupdir}/{modeltype[:-2]}_last.weights'
        logging(f'save weights to {model_save_dir}')
        cur_model.seen = (epoch + 1) * len(train_loader.dataset)
        save_net(model_save_dir,cur_model)

def test(epoch,b_iou):

    model.eval()
    if ngpus > 1:
        cur_model = model.module
    else:
        cur_model = model
    
    anchors     = cur_model.anchors
    num_anchors = cur_model.num_anchors
    anchor_step = int(len(anchors)/num_anchors)
    total       = 0.0
    proposals   = 0.0


    for _, (data, target) in enumerate(test_loader):
        with torch.no_grad():
            if use_cuda:
                data = data.cuda()
            data = Variable(data)
            output = model(data).data
            batch = output.size(0)
            h = output.size(2)
            w = output.size(3)
            output = output.view(batch*num_anchors, 5, h*w).transpose(0,1).contiguous().view(5, batch*num_anchors*h*w)
            grid_x = torch.linspace(0, w-1, w).repeat(h,1).repeat(batch*num_anchors, 1, 1).view(batch*num_anchors*h*w).cuda()
            grid_y = torch.linspace(0, h-1, h).repeat(w,1).t().repeat(batch*num_anchors, 1, 1).view(batch*num_anchors*h*w).cuda()
            xs = torch.sigmoid(output[0]) + grid_x
            ys = torch.sigmoid(output[1]) + grid_y

            anchor_w = torch.Tensor(anchors).view(num_anchors, anchor_step).index_select(1, torch.LongTensor([0]))
            anchor_h = torch.Tensor(anchors).view(num_anchors, anchor_step).index_select(1, torch.LongTensor([1]))
            anchor_w = anchor_w.repeat(batch, 1).repeat(1, 1, h*w).view(batch*num_anchors*h*w).cuda()
            anchor_h = anchor_h.repeat(batch, 1).repeat(1, 1, h*w).view(batch*num_anchors*h*w).cuda()
            ws = torch.exp(output[2]) * anchor_w
            hs = torch.exp(output[3]) * anchor_h
            det_confs = torch.sigmoid(output[4])
            sz_hw = h*w
            sz_hwa = sz_hw*num_anchors
            det_confs = convert2cpu(det_confs)
            xs = convert2cpu(xs)
            ys = convert2cpu(ys)
            ws = convert2cpu(ws)
            hs = convert2cpu(hs)

            for b in range(batch):
                det_confs_inb = det_confs[b*sz_hwa:(b+1)*sz_hwa].numpy()
                xs_inb = xs[b*sz_hwa:(b+1)*sz_hwa].numpy()
                ys_inb = ys[b*sz_hwa:(b+1)*sz_hwa].numpy()
                ws_inb = ws[b*sz_hwa:(b+1)*sz_hwa].numpy()
                hs_inb = hs[b*sz_hwa:(b+1)*sz_hwa].numpy()
                ind = np.argmax(det_confs_inb)

                bcx = xs_inb[ind]
                bcy = ys_inb[ind]
                bw = ws_inb[ind]
                bh = hs_inb[ind]

                box = [bcx/w, bcy/h, bw/w, bh/h]

                iou = bbox_iou(box, target[b][1:5], x1y1x2y2=False)
                proposals = proposals + iou
                total = total+1
        

    avg_ious = proposals/total
    logging("iou: %f, best iou: %f" % (avg_ious,b_iou))
    if avg_ious>b_iou:
        b_iou = avg_ious
        save_net(f'{backupdir}/{modeltype[:-2]}_best_test.weights', cur_model)
        write_accuracy(str(avg_ious.data))
    return b_iou

bit_precision = 6 #QAT training
evaluate = False

if __name__ == '__main__':
    # evaluate = True
    if evaluate:
        logging('evaluating ...')
        test(0,0)
    else:
        for epoch in range(init_epoch, max_epochs):
            if epoch%2==0:  # 每 10 个epoch，pruning 一下.把参数调整在[-1,1]之间
                    print("begin pruning...")
                    minimum = 1.0 / 2 ** bit_precision
                    for k, v in model.state_dict().items():
                        data=v.cpu().numpy()
                        for x in np.nditer(data, op_flags=['readwrite']):
                            if x[...]>1:
                                x[...]=1

                            if x[...]<-1:
                                x[...]=-1
                        param = torch.from_numpy(data)
                        v.copy_(param)
                        x[...] = round(x[...] / minimum) * minimum

            train(epoch)
            best_iou = test(epoch,best_iou)