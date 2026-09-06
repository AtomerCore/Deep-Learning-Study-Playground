import torch
import torchvision
from torch import nn
from PIL import Image
import time
import numpy as np
from torch.nn import functional as F
from matplotlib import pyplot as plt


def set_figsize(figsize=(3.5, 2.5)):
    """设置 matplotlib 图像大小"""
    plt.rcParams['figure.figsize'] = figsize


def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5):
    """绘制图像列表"""
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten()

    for i, (ax, img) in enumerate(zip(axes, imgs)):
        if torch.is_tensor(img):
            ax.imshow(img.numpy())
        else:
            ax.imshow(img)

        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)

        if titles:
            ax.set_title(titles[i])

    plt.show()
    return axes

# 前置：实现累加器
def accuracy(y_hat, y):
    """计算预测正确的样本数量（返回浮点数）。"""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.sum())


# 前置：实现正确预测样本数
class Accumulator:
    """在n个变量上进行累加。"""
    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def __getitem__(self, idx):
        return self.data[idx]


# 前置：定义Timer
class Timer:  #@save
    """记录多次运行时间。"""
    def __init__(self):
        """初始化，创建一个空的时间列表，并立即开始计时。"""
        self.times = []      # 用于存储每次计时的结果（秒）
        self.start()         # 调用start()方法开始计时

    def start(self):
        """启动计时器。"""
        self.tik = time.time()  # 记录当前时间作为开始时间

    def stop(self):
        """停止计时器，将本次耗时记录到times列表中。"""
        self.times.append(time.time() - self.tik)  # 计算耗时并存入列表
        return self.times[-1]  # 返回刚记录的本次耗时

    def avg(self):
        """返回所有记录时间的平均值。"""
        return sum(self.times) / len(self.times)

    def sum(self):
        """返回所有记录时间的总和。"""
        return sum(self.times)

    def cumsum(self):
        """返回所有记录时间的累计值列表。"""
        return np.array(self.times).cumsum().tolist()


# 前置：定义一个在动画中绘制数据的实用程序类
class Animator:
    """在动画中绘制数据。"""
    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None,
                 ylim=None, xscale='linear', yscale='linear',
                 fmts=('-', 'm--', 'g-.', 'r:'), nrows=1, ncols=1,
                 figsize=(3.5, 2.5)):
        if legend is None:
            legend = []

        self.fig, self.axes = plt.subplots(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1:
            self.axes = [self.axes,]

        def config_axes():
            ax = self.axes[0]
            ax.set_xlabel(xlabel) if xlabel else None
            ax.set_ylabel(ylabel) if ylabel else None
            if xlim: ax.set_xlim(xlim)
            if ylim: ax.set_ylim(ylim)
            ax.set_xscale(xscale)
            ax.set_yscale(yscale)
            if legend: ax.legend(legend)

        self.config_axes = config_axes

        self.X, self.Y, self.fmts = None, None, fmts
        plt.ion()

    def add(self, x, y):
        if not hasattr(y, "__len__"):
            y = [y]
        n = len(y)
        if not hasattr(x, "__len__"):
            x = [x] * n
        if not self.X:
            self.X = [[] for _ in range(n)]
        if not self.Y:
            self.Y = [[] for _ in range(n)]
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
        self.axes[0].cla()
        for x, y, fmt in zip(self.X, self.Y, self.fmts):
            self.axes[0].plot(x, y, fmt)
        self.config_axes()

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.01)

    # 训练结束后显示最终图表
    def show(self):
        plt.ioff()  # 关闭交互模式
        plt.show()
        plt.pause(1)


# 前置：定义evaluate_accuracy_gpu
def evaluate_accuracy_gpu(net, data_iter, device=None):
    """使用GPU计算模型在数据集上的精度。"""
    if isinstance(net, torch.nn.Module):
        net.eval()
        if not device:
            device = next(iter(net.parameters())).device
    metric = Accumulator(2)
    for X, y in data_iter:
        if isinstance(X, list):
            X = [x.to(device) for x in X]
        else:
            X = X.to(device)
        y = y.to(device)
        metric.add(accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]

# 这两个函数允许我们在请求的GPU不存在的情况下运行代码
def try_gpu(i=0):
    """如果存在，则返回gpu(i)，否则返回cpu()。"""
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')


def try_all_gpus():
    """返回所有可用的GPU，如果没有GPU，则返回[cpu(),]。"""
    devices = [
        torch.device(f'cuda:{i}') for i in range(torch.cuda.device_count())]
    return devices if devices else [torch.device('cpu')]


# 前置：残差块
class Residual(nn.Module):
    def __init__(self, input_channels, num_channels,
                 use_1x1conv=False, strides=1):
        super().__init__()

        self.conv1 = nn.Conv2d(
            input_channels,
            num_channels,
            kernel_size=3,
            padding=1,
            stride=strides
        )

        self.conv2 = nn.Conv2d(
            num_channels,
            num_channels,
            kernel_size=3,
            padding=1
        )

        if use_1x1conv:
            self.conv3 = nn.Conv2d(
                input_channels,
                num_channels,
                kernel_size=1,
                stride=strides
            )
        else:
            self.conv3 = None

        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))

        if self.conv3:
            X = self.conv3(X)

        Y += X

        return F.relu(Y)


# 前置：resnet18模型
def resnet18(num_classes, in_channels=1):
    """稍作修改的ResNet-18模型"""

    def resnet_block(in_channels, out_channels,
                     num_residuals, first_block=False):

        blk = []

        for i in range(num_residuals):
            if i == 0 and not first_block:
                blk.append(
                    Residual(
                        in_channels,
                        out_channels,
                        use_1x1conv=True,
                        strides=2
                    )
                )
            else:
                blk.append(
                    Residual(
                        out_channels,
                        out_channels
                    )
                )

        return nn.Sequential(*blk)

    net = nn.Sequential(
        nn.Conv2d(
            in_channels,
            64,
            kernel_size=3,
            stride=1,
            padding=1
        ),
        nn.BatchNorm2d(64),
        nn.ReLU()
    )

    net.add_module(
        "resnet_block1",
        resnet_block(64, 64, 2, first_block=True)
    )

    net.add_module(
        "resnet_block2",
        resnet_block(64, 128, 2)
    )

    net.add_module(
        "resnet_block3",
        resnet_block(128, 256, 2)
    )

    net.add_module(
        "resnet_block4",
        resnet_block(256, 512, 2)
    )

    net.add_module(
        "global_avg_pool",
        nn.AdaptiveAvgPool2d((1, 1))
    )

    net.add_module(
        "fc",
        nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, num_classes)
        )
    )

    return net







# 显示图像
set_figsize()

img = Image.open("./img/cat1.png")
plt.imshow(img)
plt.show()


# 对同一张图片重复执行多次随机图像增广，并将结果按网格展示出来
def apply(img, aug, num_rows=2, num_cols=4, scale=1.5):
    Y = [aug(img) for _ in range(num_rows * num_cols)]
    show_images(Y, num_rows, num_cols, scale=scale)


# 左右翻转图像
apply(img, torchvision.transforms.RandomHorizontalFlip())


# 上下反转图像
apply(img, torchvision.transforms.RandomVerticalFlip())


# 随机裁剪
shape_aug = torchvision.transforms.RandomResizedCrop(
    (200, 200), scale=(0.1, 1), ratio=(0.5, 2))
apply(img, shape_aug)


# 随机更改图像的亮度
apply(img,
      torchvision.transforms.ColorJitter(brightness=0.5, contrast=0,saturation=0, hue=0)
      )


# 随机更改图像的色调
apply(img,
      torchvision.transforms.ColorJitter(brightness=0, contrast=0, saturation=0,hue=0.5)
      )


# 随机更改图像的亮度（brightness）、对比度（contrast）、饱和度（saturation）和色调（hue）
color_aug = torchvision.transforms.ColorJitter(brightness=0.5, contrast=0.5,
                                               saturation=0.5, hue=0.5)
apply(img, color_aug)


# 结合多种图像增广方法
augs = torchvision.transforms.Compose([
    torchvision.transforms.RandomHorizontalFlip(), color_aug, shape_aug])
apply(img, augs)



# 使用图像增广进行训练

# 下载并展示数据集
all_images = torchvision.datasets.CIFAR10(train=True, root="./data",
                                          download=True)
show_images([all_images[i][0] for i in range(32)], 4, 8, scale=0.8);

# 只使用最简单的随机左右翻转
train_augs = torchvision.transforms.Compose([
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor()])

test_augs = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()])

# 定义一个辅助函数，以便于读取图像和应用图像增广
def load_cifar10(is_train, augs, batch_size):
    dataset = torchvision.datasets.CIFAR10(root="./data", train=is_train,
                                           transform=augs, download=True)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                             shuffle=is_train, num_workers=0)
    return dataloader

# 定义一个函数，使用多GPU对模型进行训练和评估
def train_batch_ch13(net, X, y, loss, trainer, devices):
    if isinstance(X, list):
        X = [x.to(devices[0]) for x in X]
    else:
        X = X.to(devices[0])
    y = y.to(devices[0])
    net.train()
    trainer.zero_grad()
    pred = net(X)
    l = loss(pred, y)
    l.sum().backward()
    trainer.step()
    train_loss_sum = l.sum()
    train_acc_sum = accuracy(pred, y)
    return train_loss_sum, train_acc_sum

def train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs,
               devices=try_all_gpus()):
    timer, num_batches = Timer(), len(train_iter)
    animator = Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0, 1],
                            legend=['train loss', 'train acc', 'test acc'])
    net = nn.DataParallel(net, device_ids=devices).to(devices[0])
    for epoch in range(num_epochs):
        metric = Accumulator(4)
        for i, (features, labels) in enumerate(train_iter):
            timer.start()
            l, acc = train_batch_ch13(net, features, labels, loss, trainer,
                                      devices)
            metric.add(l, acc, labels.shape[0], labels.numel())
            timer.stop()
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(
                    epoch + (i + 1) / num_batches,
                    (metric[0] / metric[2], metric[1] / metric[3], None))
        test_acc = evaluate_accuracy_gpu(net, test_iter)
        animator.add(epoch + 1, (None, None, test_acc))
    print(f'loss {metric[0] / metric[2]:.3f}, train acc '
          f'{metric[1] / metric[3]:.3f}, test acc {test_acc:.3f}')
    print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec on '
          f'{str(devices)}')
    animator.show()

# 定义 train_with_data_aug 函数，使用图像增广来训练模型
batch_size, devices, net = 256, try_all_gpus(), resnet18(10, 3)

def init_weights(m):
    if type(m) in [nn.Linear, nn.Conv2d]:
        nn.init.xavier_uniform_(m.weight)

net.apply(init_weights)

def train_with_data_aug(train_augs, test_augs, net, lr=0.001):
    train_iter = load_cifar10(True, train_augs, batch_size)
    test_iter = load_cifar10(False, test_augs, batch_size)
    loss = nn.CrossEntropyLoss(reduction="none")
    trainer = torch.optim.Adam(net.parameters(), lr=lr)
    train_ch13(net, train_iter, test_iter, loss, trainer, 10, devices)

# 训练模型
train_with_data_aug(train_augs, test_augs, net)

