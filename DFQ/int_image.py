from PIL import Image
import numpy as np
import torch
import h5py
from modeling.detection.skynet import *
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import os

def convert2binary(pic_path):
    with open('conv0.bb', 'wb') as f:
        for img_index in range(12):
            pic_name = pic_path + str(img_index) + '.jpg'
            img = Image.open(pic_name).convert('RGBA').resize((320, 160))
            img = np.array(img)
            for c in range(4):
                for h in range(160):
                    for w in range(320):
                        pixel = img[h][w][c]
                        byte = bytes([pixel])
                        f.write(byte)
    f.close()


class DACDataset(Dataset):
    def __init__(self, root, shape=None, transform=None, batch_size=32, num_workers=4):
        self.files = [file for file in os.listdir(root) if os.path.isfile(os.path.join(root, file))]
        self.imageNames = [file.split('.')[0] for file in self.files]
        self.files = [os.path.join(root, file) for file in self.files]
        self.nSamples = len(self.files)
        self.transform = transform
        self.shape = shape
        self.batch_size = batch_size
        self.num_workers = num_workers

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        imgpath = self.files[index]
        img = Image.open(imgpath).convert('RGB')
        if self.shape:
            img = img.resize(self.shape)

        if self.transform is not None:
            img = self.transform(img)

        return img


def load_net(fname, net):
    with h5py.File(fname, 'r') as h5f:
        for k, v in net.state_dict().items():
            param = torch.from_numpy(np.asarray(h5f[k]))
            v.copy_(param)


def convert2cpu(gpu_matrix):
    return torch.FloatTensor(gpu_matrix.size()).copy_(gpu_matrix)


def run1image():
    net = SkyNet()
    load_net('../../Training/train_result/tempSkyNet().weights', net)
    net.eval()

    batch_size = 1
    init_width = net.width
    init_height = net.height
    dataset = DACDataset('SAR-eval/', shape=(init_width, init_height),
                         transform=transforms.Compose([
                             transforms.ToTensor(),
                         ]))
    test_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=10, pin_memory=True)

    anchors = net.anchors
    num_anchors = net.num_anchors
    anchor_step = len(anchors) // num_anchors
    h = 20
    w = 40
    total = 0
    imageNum = 1
    results = np.zeros((imageNum, 5))

    grid_x = torch.linspace(0, w - 1, w).repeat(h, 1).repeat(batch_size * num_anchors, 1, 1).view(
        batch_size * num_anchors * h * w)
    grid_y = torch.linspace(0, h - 1, h).repeat(w, 1).t().repeat(batch_size * num_anchors, 1, 1).view(
        batch_size * num_anchors * h * w)
    anchor_w = torch.Tensor(anchors).view(num_anchors, anchor_step).index_select(1, torch.LongTensor([0]))
    anchor_h = torch.Tensor(anchors).view(num_anchors, anchor_step).index_select(1, torch.LongTensor([1]))
    anchor_w = anchor_w.repeat(batch_size, 1).repeat(1, 1, h * w).view(batch_size * num_anchors * h * w)
    anchor_h = anchor_h.repeat(batch_size, 1).repeat(1, 1, h * w).view(batch_size * num_anchors * h * w)
    sz_hw = h * w
    sz_hwa = sz_hw * num_anchors

    for batch_idx, data in enumerate(test_loader):
        data = data
        output = net(data).data
        batch = output.size(0)
        output = output.view(batch * num_anchors, 5, h * w).transpose(0, 1).contiguous().view(5,batch * num_anchors * h * w)
        det_confs = torch.sigmoid(output[4])

        for b in range(batch):
            det_confs_inb = det_confs[b * sz_hwa:(b + 1) * sz_hwa].numpy()
            ind = np.argmax(det_confs_inb)

            xs_inb = torch.sigmoid(output[0, b * sz_hwa + ind]) + grid_x[b * sz_hwa + ind]
            ys_inb = torch.sigmoid(output[1, b * sz_hwa + ind]) + grid_y[b * sz_hwa + ind]
            ws_inb = torch.exp(output[2, b * sz_hwa + ind]) * anchor_w[b * sz_hwa + ind]
            hs_inb = torch.exp(output[3, b * sz_hwa + ind]) * anchor_h[b * sz_hwa + ind]

            bcx = xs_inb.item() / w
            bcy = ys_inb.item() / h
            bw = ws_inb.item() / w
            bh = hs_inb.item() / h

            xmin = bcx - bw / 2.0
            ymin = bcy - bh / 2.0
            xmax = xmin + bw
            ymax = ymin + bh

            print([xmin, xmax, ymin, ymax])
            print([xmin * 320, xmax * 320, ymin * 160, ymax * 160])

'''
用于为HLS产生一个输入图片的二进制数据
并在python中计算该图片所对应的锚框
'''
if __name__ == '__main__':
    print(os.getcwd())
    # convert2binary('../../Training/train_result/img_eva/Gao_ship_hh_0201608254401010021.jpg')
    convert2binary('img_test/')
    # run1image()
