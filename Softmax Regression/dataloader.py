import torch
import torchvision
from torch.utils import data
from torchvision import transforms
import matplotlib.pyplot as plt
import time

class Timer:
    def __init__(self):
        self.start()

    def start(self):
        self.tik = time.time()

    def stop(self):
        return time.time() - self.tik


def get_fashion_mnist_labels(labels):
    text_labels = [
        't-shirt', 'trouser', 'pullover', 'dress', 'coat', 'sandal', 'shirt',
        'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]


def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5):
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


# ─────────────────────────────────────────────


if __name__ == '__main__':
    # 1. 初始化
    trans = transforms.ToTensor()

    # 2. 加载数据集
    mnist_train = torchvision.datasets.FashionMNIST(
        root="./data", train=True, transform=trans, download=True)
    mnist_test = torchvision.datasets.FashionMNIST(
        root="./data", train=False, transform=trans, download=True)

    # 3. 基础测试
    print(len(mnist_train), len(mnist_test))
    print(mnist_train[0][0].shape)

    # 4. 图片显示测试
    X, y = next(iter(data.DataLoader(mnist_train, batch_size=18)))
    show_images(X.reshape(18, 28, 28), 2, 9, titles=get_fashion_mnist_labels(y))
    plt.show()

    # 5. DataLoader 速度测试（现在用 num_workers=0，安全）
    batch_size = 256
    train_iter = data.DataLoader(mnist_train, batch_size, shuffle=True,
                                 num_workers=get_dataloader_workers())
    timer = Timer()
    for X, y in train_iter:
        continue
    print(f'{timer.stop():.2f} sec')

    # 6. Resize 功能测试
    train_iter, test_iter = load_data_fashion_mnist(32, resize=64)
    for X, y in train_iter:
        print(X.shape, X.dtype, y.shape, y.dtype)
        break

    print("Successfully!")
