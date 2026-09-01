import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torch.nn import functional as F
import matplotlib.pyplot as plt
import numpy as np
import time
from torch import nn
import pandas as pd
from PIL import Image
import os


# 自定义数据集
class LeafDataset(Dataset):
    def __init__(self, df, img_dir, transform=None, label_map=None):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform
        self.label_map = label_map  # str -> int 的映射字典

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = str(self.df.iloc[idx, 0])
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)


        if 'label' in self.df.columns and self.label_map is not None:
            label_str = self.df.iloc[idx, 1]
            label = self.label_map[label_str]
            return image, label
        else:
            return image, img_name


# 加载数据集
def load_data_custom(batch_size, resize=96):
    """加载 Classify Leaves，自动 8:2 划分 train/val"""
    data_root = './data/classify-leaves'
    train_csv = os.path.join(data_root, 'train.csv')

    df = pd.read_csv(train_csv)

    # 标签编码：字符串 到 整数
    unique_labels = sorted(df['label'].unique())
    label_map = {name: i for i, name in enumerate(unique_labels)}
    num_classes = len(unique_labels)
    print(f'检测到 {num_classes} 个类别')

    # 切分比例8:2
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    n_train = int(0.8 * len(df))
    df_train = df.iloc[:n_train]
    df_val   = df.iloc[n_train:]

    trans_train = transforms.Compose([
        transforms.Resize((resize, resize)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    trans_val = transforms.Compose([
        transforms.Resize((resize, resize)),
        transforms.ToTensor(),
    ])


    train_ds = LeafDataset(df_train, data_root, trans_train, label_map)
    val_ds   = LeafDataset(df_val,   data_root, trans_val,   label_map)

    train_iter = DataLoader(train_ds, batch_size, shuffle=True,  num_workers=0)
    val_iter   = DataLoader(val_ds,   batch_size, shuffle=False, num_workers=0)

    return train_iter, val_iter, num_classes, label_map


# 前置代码

def accuracy(y_hat, y):
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.sum())

class Accumulator:
    def __init__(self, n):
        self.data = [0.0] * n
    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]
    def __getitem__(self, idx):
        return self.data[idx]

class Timer:
    def __init__(self):
        self.times = []
        self.start()
    def start(self):
        self.tik = time.time()
    def stop(self):
        self.times.append(time.time() - self.tik)
        return self.times[-1]
    def avg(self):
        return sum(self.times) / len(self.times)
    def sum(self):
        return sum(self.times)
    def cumsum(self):
        return np.array(self.times).cumsum().tolist()

class Animator:
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
    def show(self):
        plt.ioff()
        plt.show()
        plt.pause(1)

def evaluate_accuracy_gpu(net, data_iter, device=None):
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

def train_ch6(net, train_iter, test_iter, num_epochs, lr, device):
    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)
    net.apply(init_weights)
    print('training on', device)
    net.to(device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=0.01)
    loss = nn.CrossEntropyLoss(label_smoothing=0.1)
    animator = Animator(xlabel='epoch', xlim=[1, num_epochs],
                            legend=['train loss', 'train acc', 'test acc'])
    timer, num_batches = Timer(), len(train_iter)
    for epoch in range(num_epochs):
        metric = Accumulator(3)
        net.train()
        for i, (X, y) in enumerate(train_iter):
            timer.start()
            optimizer.zero_grad()
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            l.backward()
            optimizer.step()
            with torch.no_grad():
                metric.add(l * X.shape[0], accuracy(y_hat, y), X.shape[0])
            timer.stop()
            train_l = metric[0] / metric[2]
            train_acc = metric[1] / metric[2]
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches,
                             (train_l, train_acc, None))
        test_acc = evaluate_accuracy_gpu(net, test_iter)
        animator.add(epoch + 1, (None, None, test_acc))
    print(f'loss {train_l:.3f}, train acc {train_acc:.3f}, 'f'test acc {test_acc:.3f}')
    print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec 'f'on {str(device)}')
    animator.show()

def try_gpu(i=0):
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')




# ResNet
class Residual(nn.Module):
    def __init__(self, input_channels, num_channels,
                 use_1x1conv=False, strides=1, dropout=0.3):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, num_channels,
                               kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.Conv2d(num_channels, num_channels,
                               kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(input_channels, num_channels,
                                   kernel_size=1, stride=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        Y = self.dropout(Y)
        if self.conv3:
            X = self.conv3(X)
        Y += X
        return F.relu(Y)

# 通道数为3 (RGB)
b1 = nn.Sequential(nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
                   nn.BatchNorm2d(64), nn.ReLU(),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

def resnet_block(input_channels, num_channels, num_residuals,
                 first_block=False):
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(Residual(input_channels, num_channels,
                                use_1x1conv=True, strides=2))
        elif i == 0 and first_block:
            if input_channels != num_channels:
                blk.append(Residual(input_channels, num_channels,
                                    use_1x1conv=True, strides=1))
            else:
                blk.append(Residual(input_channels, num_channels))
        else:
            blk.append(Residual(num_channels, num_channels))
    return blk

b2 = nn.Sequential(*resnet_block(64, 128, 3, first_block=True))
b3 = nn.Sequential(*resnet_block(128, 256, 3))




lr, num_epochs, batch_size = 1e-3, 50, 128
train_iter, test_iter, num_classes, label_map = load_data_custom(batch_size, resize=96)


net = nn.Sequential(
    b1, b2, b3,
    nn.AdaptiveAvgPool2d((1, 1)),
    nn.Flatten(),
    nn.Dropout(0.5),
    nn.Linear(256, num_classes)
)

train_ch6(net, train_iter, test_iter, num_epochs, lr, try_gpu())




# 生成 submission.csv
def predict_and_save(net, label_map, device, resize=96):
    """读取 test.csv，预测并生成提交文件"""
    data_root = './data/classify-leaves'
    test_csv = os.path.join(data_root, 'test.csv')
    df_test = pd.read_csv(test_csv)

    # 数字 到 字符串
    inv_label_map = {v: k for k, v in label_map.items()}

    trans = transforms.Compose([
        transforms.Resize((resize, resize)),
        transforms.ToTensor(),
    ])
    test_ds = LeafDataset(df_test, data_root, trans, label_map=None)
    test_loader = DataLoader(test_ds, batch_size, shuffle=False, num_workers=0)

    net.eval()
    net.to(device)
    results = []

    with torch.no_grad():
        for X, names in test_loader:
            X = X.to(device)
            preds = net(X).argmax(dim=1)
            for name, pred in zip(names, preds.cpu().numpy()):
                results.append((name, inv_label_map[pred]))

    df_sub = pd.DataFrame(results, columns=['image', 'label'])
    df_sub.to_csv('submission.csv', index=False)
    print('已保存 submission.csv')

predict_and_save(net, label_map, try_gpu(), resize=96)
