import torch

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

# 实现多输入通道互相关运算
def corr2d_multi_in(X, K):
    return sum(corr2d(x, k) for x, k in zip(X, K))


# 验证互相关运算的输出
X = torch.tensor([[[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]],
                  [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]])
K = torch.tensor([[[0.0, 1.0], [2.0, 3.0]], [[1.0, 2.0], [3.0, 4.0]]])

print(corr2d_multi_in(X, K))


# 计算多个通道的输出的互相关函数
def corr2d_multi_in_out(X, K):
    return torch.stack([corr2d_multi_in(X, k) for k in K], 0)

K = torch.stack((K, K + 1, K + 2), 0)
print(K.shape)
print(corr2d_multi_in_out(X, K))