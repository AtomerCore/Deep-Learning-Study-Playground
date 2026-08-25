import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import time
from torch import nn


# 前置：加载数据集
def load_data_fashion_mnist(batch_size, resize=None):
    """下载Fashion-MNIST数据集，然后将其加载到内存中。"""
    # 图像预处理
    trans = [transforms.ToTensor()]  # 将图像转换为PyTorch张量，并自动将像素值从0-255缩放到0-1
    if resize:
        trans.insert(0, transforms.Resize(resize))  # 如果指定了resize，则插入调整大小操作
    trans = transforms.Compose(trans)

    # 下载并加载训练集和测试集
    # 如果root目录下没有数据，download=True会自动下载
    mnist_train = datasets.FashionMNIST(root='./data', train=True, transform=trans, download=True)
    mnist_test = datasets.FashionMNIST(root='./data', train=False, transform=trans, download=True)

    # 创建DataLoader以按批次读取数据
    return (DataLoader(mnist_train, batch_size, shuffle=True, num_workers=0),
            DataLoader(mnist_test, batch_size, shuffle=False, num_workers=0))


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


# 前置：使用GPU训练模型（包含前面定义的Animator动画）
def train_ch6(net, train_iter, test_iter, num_epochs, lr, device):
    """用GPU训练模型(在第六章定义)"""
    def init_weights(m):
        # 只对 nn.Linear 和 nn.Conv2d 做 Xavier 均匀初始化
        # 原理见之前"数值稳定性"一讲：根据输入输出大小让随机输入时输出方差差不多，保证模型开始训练时不炸
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)
    net.apply(init_weights)  # 递归地把该函数应用到网络每一个子模块上

    print('training on', device)  # 打印设备——常见错误是以为在 GPU 上跑实际没有，半天不出结果，所以一定要打出来确认
    net.to(device)  # 把整个网络的参数搬到 GPU

    optimizer = torch.optim.SGD(net.parameters(), lr=lr)  # 优化器就用普通 SGD（给个学习率即可，不用花哨的）
    loss = nn.CrossEntropyLoss()  # 损失用 CrossEntropyLoss（多类分类，跟 softmax 回归一样）
    animator = Animator(xlabel='epoch', xlim=[1, num_epochs],
                            legend=['train loss', 'train acc', 'test acc'])
    timer, num_batches = Timer(), len(train_iter)

    for epoch in range(num_epochs):
        # 训练损失之和、训练准确率之和、样本数
        metric = Accumulator(3)
        net.train()
        for i, (X, y) in enumerate(train_iter):
            timer.start()
            optimizer.zero_grad()
            X, y = X.to(device), y.to(device)  # 核心区别：把 X、y 挪到 GPU
            y_hat = net(X)
            l = loss(y_hat, y)
            l.backward()
            optimizer.step()
            with torch.no_grad():
                metric.add(l * X.shape[0], accuracy(y_hat, y), X.shape[0])  # 累加指标（不做梯度计算）
            timer.stop()
            train_l = metric[0] / metric[2]
            train_acc = metric[1] / metric[2]
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                # 每 num_batches//5 个 batch 画一次中间曲线（动画效果）
                animator.add(epoch + (i + 1) / num_batches,
                             (train_l, train_acc, None))
        test_acc = evaluate_accuracy_gpu(net, test_iter)
        # 每个 epoch 结束在测试集上算一次精度并画图
        animator.add(epoch + 1, (None, None, test_acc))
    print(f'loss {train_l:.3f}, train acc {train_acc:.3f}, 'f'test acc {test_acc:.3f}')
    # 最后打印 loss/训练精度/测试精度 和 examples/sec 吞吐量
    print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec 'f'on {str(device)}')
    animator.show()

# 前置：尝试获取GPU
def try_gpu(i=0):
    """如果存在，则返回gpu(i)，否则返回cpu()。"""
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')





# 定义VGG块
def vgg_block(num_convs, in_channels, out_channels):
    layers = []
    for _ in range(num_convs):
        layers.append(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
        layers.append(nn.ReLU())
        in_channels = out_channels
    layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
    return nn.Sequential(*layers)


# 定义VGG网络
conv_arch = ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))

def vgg(conv_arch):
    conv_blks = []
    in_channels = 1
    for (num_convs, out_channels) in conv_arch:
        conv_blks.append(vgg_block(num_convs, in_channels, out_channels))
        in_channels = out_channels

    return nn.Sequential(*conv_blks, nn.Flatten(),
                         nn.Linear(out_channels * 7 * 7, 4096), nn.ReLU(),
                         nn.Dropout(0.5), nn.Linear(4096, 4096), nn.ReLU(),
                         nn.Dropout(0.5), nn.Linear(4096, 10))

net = vgg(conv_arch)


# 观察每个层输出的形状
X = torch.randn(size=(1, 1, 224, 224))
for blk in net:
    X = blk(X)
    print(blk.__class__.__name__, 'output shape:\t', X.shape)


# 由于VGG-11比AlexNet计算量更大，因此构建一个通道数较少的网络
ratio = 4
small_conv_arch = [(pair[0], pair[1] // ratio) for pair in conv_arch]
net = vgg(small_conv_arch)


# 模型训练
lr, num_epochs, batch_size = 0.05, 10, 128
train_iter, test_iter = load_data_fashion_mnist(batch_size, resize=224)
train_ch6(net, train_iter, test_iter, num_epochs, lr, try_gpu())





