import torch
from IPython import display
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
from dataloader import get_dataloader_workers
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


def sgd(params, lr, batch_size):
    """小批量随机梯度下降"""
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()




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




# 实现 softmax
def softmax(X):
    X_exp = torch.exp(X)
    partition = X_exp.sum(1, keepdim=True)
    return X_exp / partition




# 实现 softmax 回归模型
def net(X):
    return softmax(torch.matmul(X.reshape((-1, W.shape[0])), W) + b)



# 实现交叉熵损失函数
def cross_entropy(y_hat, y):
    return -torch.log(y_hat[range(len(y_hat)), y])



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


# 小批量随机梯度下降来优化模型的损失函数
lr = 0.1
def updater(batch_size):
    return sgd([W, b], lr, batch_size)

# 对图像进行分类预测
def predict_ch3(net, test_iter, n=6):
    """预测标签（定义见第 3 章）。"""
    for X, y in test_iter:
        break
    trues = get_fashion_mnist_labels(y)
    preds = get_fashion_mnist_labels(net(X).argmax(axis=1))
    titles = [true + '\n' + pred for true, pred in zip(trues, preds)]
    show_images(X[0:n].reshape((n, 28, 28)), 1, n, titles=titles[0:n])


# 执行代码在main中
if __name__ == '__main__':

    batch_size = 256
    train_iter, test_iter = load_data_fashion_mnist(batch_size)

    # 将展平每个图像，把它们看作长度为 784 的向量。因为我们的数据集有 10 个类别，所以网络输出维度为 10
    num_inputs = 784
    num_outputs = 10

    # 我们将每个元素变成一个非负数。此外，依据概率原理，每行总和为 1
    W = torch.normal(0, 0.01, size=(num_inputs, num_outputs), requires_grad=True)
    b = torch.zeros(num_outputs, requires_grad=True)

    # 测试张量求和
    # 给定一个矩阵 X，我们可以对所有元素求和
    X = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    print("X.sum(0, keepdim=True):", X.sum(0, keepdim=True))
    print("X.sum(1, keepdim=True):", X.sum(1, keepdim=True))


    # 基础功能验证
    print("=== 基础函数验证 ===")
    X_test = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    print(f"X.sum(0): {X_test.sum(0, keepdim=True)}")
    print(f"X.sum(1): {X_test.sum(1, keepdim=True)}")

    X_prob = softmax(torch.normal(0, 1, (2, 5)))
    print(f"softmax 行和：{X_prob.sum(1)}")

    # 创建一个数据 y^，其中包含 2 个样本在 3 个类别的预测概率，使用 y 作为 y^中概率的索引
    y = torch.tensor([0, 2])
    y_hat = torch.tensor([[0.1, 0.3, 0.6], [0.3, 0.2, 0.5]])
    print(f"交叉熵：{cross_entropy(y_hat, y)}")
    print(f"准确率：{accuracy(y_hat, y) / len(y)}")


    print(f"\n=== 未训练模型评估 ===")
    initial_acc = evaluate_accuracy(net, test_iter)
    print(f"初始测试准确率：{initial_acc * 100:.2f}%")


    # 训练模型 10 个迭代周期并预测
    num_epochs = 10
    train_ch3(net, train_iter, test_iter, cross_entropy, num_epochs, updater)

    final_acc = evaluate_accuracy(net, test_iter)
    print(f"训练完成 - 最终测试准确率：{final_acc * 100:.2f}%")

    print("Successfully！")
