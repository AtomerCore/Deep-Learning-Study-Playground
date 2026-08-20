import torch
from torch import nn
from IPython import display
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
from torch.utils import data


def get_fashion_mnist_labels(labels):
    """返回 Fashion-MNIST 数据集的文本标签。"""
    text_labels = [
        't-shirt', 'trouser', 'pullover', 'dress', 'coat', 'sandal', 'shirt',
        'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]


def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5):
    """Plot a list of images."""
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
    return axes


def get_dataloader_workers():
    return 4


# 前文定义过的 load_data_fashion_mnist 函数
def load_data_fashion_mnist(batch_size, resize=None):
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)
    mnist_train = torchvision.datasets.FashionMNIST(
        root="./data", train=True, transform=trans, download=True)
    mnist_test = torchvision.datasets.FashionMNIST(
        root="./data", train=False, transform=trans, download=True)
    return (
        data.DataLoader(mnist_train, batch_size, shuffle=True,
                        num_workers=get_dataloader_workers()),
        data.DataLoader(mnist_test, batch_size, shuffle=False,
                        num_workers=get_dataloader_workers())
    )


# 简单绘图函数（替代 d2l.plot）
def plot(X, Y=None, xlabel=None, ylabel=None, legend=None, xlim=None,
         ylim=None, xscale='linear', yscale='linear',
         fmts=('-', 'm--', 'g-.', 'r:'), figsize=(3.5, 2.5)):
    """绘制数据点。"""
    plt.figure(figsize=figsize)
    if Y is None:
        Y = [X]
        X = list(range(len(Y[0])))
    if not isinstance(Y, (list, tuple)):
        Y = [Y]
    for i, y in enumerate(Y):
        plt.plot(X, y, fmts[i] if i < len(fmts) else fmts[-1])
    plt.xlabel(xlabel) if xlabel else None
    plt.ylabel(ylabel) if ylabel else None
    plt.xlim(xlim) if xlim else None
    plt.ylim(ylim) if ylim else None
    plt.xscale(xscale)
    plt.yscale(yscale)
    if legend:
        plt.legend(legend)
    plt.show()


# 将预测类别与真实 y 元素进行比较
def accuracy(y_hat, y):
    """计算预测正确的数量。"""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())


# Accumulator 实例中创建了 2 个变量，用于分别存储正确预测的数量和预测的总数量
class Accumulator:
    """在`n`个变量上累加。"""
    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


# 我们可以评估在任意模型 net 的准确率
def evaluate_accuracy(net, data_iter):
    """计算在指定数据集上模型的精度。"""
    if isinstance(net, torch.nn.Module):
        net.eval()
    metric = Accumulator(2)
    for X, y in data_iter:
        metric.add(accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]


# Softmax 回归的训练
def train_epoch_ch3(net, train_iter, loss, updater):
    """训练模型一个迭代周期（定义见第 3 章）。"""
    if isinstance(net, torch.nn.Module):
        net.train()
    metric = Accumulator(3)
    for X, y in train_iter:
        y_hat = net(X)
        l = loss(y_hat, y)
        if isinstance(updater, torch.optim.Optimizer):
            updater.zero_grad()
            l.backward()
            updater.step()
            metric.add(
                float(l) * len(y), accuracy(y_hat, y),
                y.size().numel())
        else:
            l.sum().backward()
            updater(X.shape[0])
            metric.add(l.sum().item(), accuracy(y_hat, y), y.numel())
    return metric[0] / metric[2], metric[1] / metric[2]


# 定义一个在动画中绘制数据的实用程序类
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


# 训练函数
def train_ch3(net, train_iter, test_iter, loss, num_epochs, updater):
    """训练模型（定义见第 3 章）。"""
    animator = Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0.3, 0.9],
                        legend=['train loss', 'train acc', 'test acc'])
    for epoch in range(num_epochs):
        train_metrics = train_epoch_ch3(net, train_iter, loss, updater)
        test_acc = evaluate_accuracy(net, test_iter)
        animator.add(epoch + 1, train_metrics + (test_acc,))
    train_loss, train_acc = train_metrics
    assert train_loss < 0.5, train_loss
    assert train_acc <= 1 and train_acc > 0.7, train_acc
    assert test_acc <= 1 and test_acc > 0.7, test_acc


# 对图像进行分类预测
def predict_ch3(net, test_iter, n=6):
    """预测标签（定义见第 3 章）。"""
    for X, y in test_iter:
        break
    trues = get_fashion_mnist_labels(y)
    preds = get_fashion_mnist_labels(net(X).argmax(axis=1))
    titles = [true + '\n' + pred for true, pred in zip(trues, preds)]
    show_images(X[0:n].reshape((n, 28, 28)), 1, n, titles=titles[0:n])


# 实现一个具有单隐藏层的多层感知机，它包含256个隐藏单元
num_inputs, num_outputs, num_hiddens = 784, 10, 256
W1 = nn.Parameter(
    torch.randn(num_inputs, num_hiddens, requires_grad=True) * 0.01)
b1 = nn.Parameter(torch.zeros(num_hiddens, requires_grad=True))
W2 = nn.Parameter(
    torch.randn(num_hiddens, num_outputs, requires_grad=True) * 0.01)
b2 = nn.Parameter(torch.zeros(num_outputs, requires_grad=True))
params = [W1, b1, W2, b2]


# 实现ReLU激活函数
def relu(X):
    a = torch.zeros_like(X)
    return torch.max(X, a)


# 实现我们的模型
def net(X):
    X = X.reshape((-1, num_inputs))
    H = relu(X @ W1 + b1)
    return (H @ W2 + b2)


loss = nn.CrossEntropyLoss()


# 执行代码在main中
if __name__ == '__main__':

    print("=== 多层感知机：激活函数介绍 ===")

    # ReLU提供了一种非常简单的非线性变换
    x = torch.arange(-8.0, 8.0, 0.1, requires_grad=True)
    y = torch.relu(x)
    plot(x.detach(), y.detach(), 'x', 'relu(x)', figsize=(5, 2.5))

    # relu 的导数
    y.backward(torch.ones_like(x), retain_graph=True)
    plot(x.detach(), x.grad, 'x', 'grad of relu', figsize=(5, 2.5))

    # 对于一个定义域在R中的输入，sigmoid函数将输入变换为区间(0, 1)上的输出
    y = torch.sigmoid(x)
    plot(x.detach(), y.detach(), 'x', 'sigmoid(x)', figsize=(5, 2.5))

    # sigmoid 的导数
    x.grad.data.zero_()
    y.backward(torch.ones_like(x), retain_graph=True)
    plot(x.detach(), x.grad, 'x', 'grad of sigmoid', figsize=(5, 2.5))

    # Tanh(双曲正切)函数也能将其输入压缩转换到区间(-1, 1)上
    y = torch.tanh(x)
    plot(x.detach(), y.detach(), 'x', 'tanh(x)', figsize=(5, 2.5))

    # tanh 的导数
    x.grad.data.zero_()
    y.backward(torch.ones_like(x), retain_graph=True)
    plot(x.detach(), x.grad, 'x', 'grad of tanh', figsize=(5, 2.5))


    print("\n=== 多层感知机的从零开始实现 ===")
    batch_size = 256
    train_iter, test_iter = load_data_fashion_mnist(batch_size)

    # 多层感知机的训练过程与softmax回归的训练过程完全相同
    num_epochs, lr = 10, 0.1
    updater = torch.optim.SGD(params, lr=lr)
    train_ch3(net, train_iter, test_iter, loss, num_epochs, updater)

    # 在一些测试数据上应用这个模型
    predict_ch3(net, test_iter)
    plt.show()

    print("Successfully！")
