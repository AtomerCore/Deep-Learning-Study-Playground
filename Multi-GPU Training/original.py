import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.nn import functional as F
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
class Timer:
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


# 前置：这两个函数允许我们在请求的GPU不存在的情况下运行代码
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

def sgd(params, lr, batch_size):
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()



# 简单网络
scale = 0.01
W1 = torch.randn(size=(20, 1, 3, 3)) * scale
b1 = torch.zeros(20)
W2 = torch.randn(size=(50, 20, 5, 5)) * scale
b2 = torch.zeros(50)
W3 = torch.randn(size=(800, 128)) * scale
b3 = torch.zeros(128)
W4 = torch.randn(size=(128, 10)) * scale
b4 = torch.zeros(10)
params = [W1, b1, W2, b2, W3, b3, W4, b4]

def lenet(X, params):
    h1_conv = F.conv2d(input=X, weight=params[0], bias=params[1])
    h1_activation = F.relu(h1_conv)
    h1 = F.avg_pool2d(input=h1_activation, kernel_size=(2, 2), stride=(2, 2))
    h2_conv = F.conv2d(input=h1, weight=params[2], bias=params[3])
    h2_activation = F.relu(h2_conv)
    h2 = F.avg_pool2d(input=h2_activation, kernel_size=(2, 2), stride=(2, 2))
    h2 = h2.reshape(h2.shape[0], -1)
    h3_linear = torch.mm(h2, params[4]) + params[5]
    h3 = F.relu(h3_linear)
    y_hat = torch.mm(h3, params[6]) + params[7]
    return y_hat

loss = nn.CrossEntropyLoss(reduction='none')


# 向多个设备分发参数
def get_params(params, device):
    new_params = [p.clone().to(device) for p in params]
    for p in new_params:
        p.requires_grad_()
    return new_params

new_params = get_params(params, try_gpu(0))
print('b1 weight:', new_params[1])
print('b1 grad:', new_params[1].grad)


# allreduce函数将所有向量相加，并将结果广播给所有GPU
def allreduce(data):
    for i in range(1, len(data)):
        data[0][:] += data[i].to(data[0].device)
    for i in range(1, len(data)):
        data[i] = data[0].to(data[i].device)

data = [torch.ones((1, 2), device=try_gpu(i)) * (i + 1) for i in range(2)]
print('before allreduce:\n', data[0], '\n', data[1])
allreduce(data)
print('after allreduce:\n', data[0], '\n', data[1])


# 将一个小批量数据均匀地分布在多个GPU上
data = torch.arange(20).reshape(4, 5)
devices = [torch.device('cuda:0'), torch.device('cuda:1')]
split = nn.parallel.scatter(data, devices)
print('input :', data)
print('load into', devices)
print('output:', split)

def split_batch(X, y, devices):
    """将`X`和`y`拆分到多个设备上"""
    assert X.shape[0] == y.shape[0]
    return (nn.parallel.scatter(X, devices), nn.parallel.scatter(y, devices))



# 在一个小批量上实现多 GPU 训练
def train_batch(X, y, device_params, devices, lr):
    X_shards, y_shards = split_batch(X, y, devices)
    ls = [
        loss(lenet(X_shard, device_W),
             y_shard).sum() for X_shard, y_shard, device_W in zip(
                 X_shards, y_shards, device_params)]
    for l in ls:
        l.backward()
    with torch.no_grad():
        for i in range(len(device_params[0])):
            allreduce([device_params[c][i].grad for c in range(len(devices))])
    for param in device_params:
        sgd(param, lr, X.shape[0])


# 定义训练函数
def train(num_gpus, batch_size, lr):
    train_iter, test_iter = load_data_fashion_mnist(batch_size)
    devices = [try_gpu(i) for i in range(num_gpus)]
    device_params = [get_params(params, d) for d in devices]
    num_epochs = 10
    animator = Animator('epoch', 'test acc', xlim=[1, num_epochs])
    timer = Timer()
    for epoch in range(num_epochs):
        timer.start()
        for X, y in train_iter:
            train_batch(X, y, device_params, devices, lr)
            torch.cuda.synchronize()
        timer.stop()
        animator.add(epoch + 1, (evaluate_accuracy_gpu(
            lambda x: lenet(x, device_params[0]), test_iter, devices[0]),))
    print(f'test acc: {animator.Y[0][-1]:.2f}, {timer.avg():.1f} sec/epoch '
          f'on {str(devices)}')


# 在单个GPU上运行
train(num_gpus=1, batch_size=256, lr=0.2)

# 增加为2个GPU
# train(num_gpus=2, batch_size=256, lr=0.2)