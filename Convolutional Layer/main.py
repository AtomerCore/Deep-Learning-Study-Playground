import torch
from torch import nn

def corr2d(X, K):
    """二维互相关运算：返回与核同样尺寸的输出"""
    h, w = K.shape                         # 读取核的高、宽
    Y = torch.zeros((X.shape[0] - h + 1,   # 输出高
                     X.shape[1] - w + 1))  # 输出宽
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            # 在输入中抠出与核同尺寸的小方块，逐元素相乘后求和
            Y[i, j] = (X[i:i+h, j:j+w] * K).sum()
    return Y



# 验证上述二维互相关运算的输出
X = torch.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]])
K = torch.tensor([[0.0, 1.0], [2.0, 3.0]])
print(corr2d(X, K))


# 实现二维卷积层
class Conv2D(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.weight = nn.Parameter(torch.rand(kernel_size))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return corr2d(x, self.weight) + self.bias


# 卷积层的一个简单应用： 检测图像中不同颜色的边缘
X = torch.ones((6, 8))
X[:, 2:6] = 0
print(X)

K = torch.tensor([[1.0, -1.0]])

"""输出Y中的1代表从白色到黑色的边缘，-1代表从黑色到白色的边缘"""
Y = corr2d(X, K)
print(Y)

# 卷积核K只可以检测垂直边缘
print(corr2d(X.t(), K))


# 学习由X生成Y的卷积核
conv2d = nn.Conv2d(1, 1, kernel_size=(1, 2), bias=False)

X = X.reshape((1, 1, 6, 8))
Y = Y.reshape((1, 1, 6, 7))

for i in range(10):
    Y_hat = conv2d(X)
    l = (Y_hat - Y)**2
    conv2d.zero_grad()
    l.sum().backward()
    conv2d.weight.data[:] -= 3e-2 * conv2d.weight.grad
    if (i + 1) % 2 == 0:
        print(f'batch {i+1}, loss {l.sum():.3f}')

# 所学的卷积核的权重张量
print(conv2d.weight.data.reshape((1, 2)))