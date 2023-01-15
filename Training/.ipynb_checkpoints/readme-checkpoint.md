## TODO

- [ ] 数据集的准备，根据 1000 张图片通过 make_data.ipynb 生成对应的 .txt 文件，可以使用 900 张用于训练，100 张用于验证
- [ ] 由于图像输入的变化 [C,H,W] 由 [3,160,320] 变为 [1,800,800]，需要对相应的代码部分做一些调整
- [ ] 训练一定的 epoch，待网络在验证集上 accuracy 收敛时，看一下 AverageIOU 是否能用，在这个过程中可使用 tensorboard 看一下 loss 曲线

## 环境

专业版 pycharm + 实验室服务器

## 数据集

Dji 

[训练集](https://pitt.app.box.com/s/756141768nn92cj0dkfbg6dan17c4h4q) [验证集](https://pitt.app.box.com/s/cq6edt2zm99s1zwa37u56gctpk1qtgpa)

SAR 舰船识别数据集

比赛官方提供

通过 make_data.ipynb 得到含有图像路径和相应标签的 .txt 文档(与自己服务器上的路径对应)，在训练时通过该文档进行图像数据和标签的读取

## 训练

### 可视化

通过 tensorboard 可实现训练过程中 loss 变化和 weight 等参数分布情况的可视化

### 参数

在 train.py 中可设置 learning_rate、epoch、batch、GPU部署、衰减率等参数，可以根据 tensorboard 中的曲线对其中一些进行调整

### 数据读取器

dataset.py 中写了 listDataset(Dataset) 类，用于训练中加载数据

其中包含 data_augmentation 的相关操作，包括 resize, crop, flip和一些图像色彩（对黑白照片这部分没必要）上的操作

### Loss

采用 region_loss.py 中自己定义的 RegionLoss 用于计算网络的损失函数

与去年比赛中的 Loss 类似，具体实现可参考相关代码，目前最多支持一张图片 15 个 bbox

### 验证

在 SkyNet 的训练中每训练一个 epoch 用验证集中 1000 张图片做测试，并根据目前网络参数在验证集上的表现决定是否对其进行记录，后面 IOU 更高的网络参数会把前面的给覆盖掉

在舰船识别数据集中只有 1000 张图片，可以将其分为 900 张训练数据，100 张验证数据，采用类似的方法进行训练和验证

### “Pruning”

该代码中还使用了 pruning 的策略，每进行十个 epoch 的训练，会对网络参数进行裁剪，在 SkyNet 中有没有这部分代码带来 accuracy 的变化不大

用了这部分代码之后，DFQ 作用其实就不明显了，因为不同通道的权重差别不会很大

在舰船识别中可用可不用，可以都试一下，比较两者 accuracy 的差异

