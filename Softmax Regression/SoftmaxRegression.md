# Softmax Regression

The Model have both [Implementation from Scratch](./softmax_original.py) and [Concise Implementation](./softmax_easy.py)

## Mermaid Map For *Implementation from Scratch*
```mermaid
graph TB
    subgraph Data_Layer["📦 数据层"]
        A1[Fashion-MNIST Dataset]
        A2[DataLoader]
        A3[transforms]
        A1 --> A2
        A3 --> A2
    end

    subgraph Model_Layer["🧠 模型层"]
        B1[输入 X<br/>784维向量]
        B2[权重 W<br/>784x10]
        B3[偏置 b<br/>10维]
        B4[softmax 函数]
        B5[输出 概率分布]
        B1 --> B4
        B2 --> B4
        B3 --> B4
        B4 --> B5
    end

    subgraph Loss_Layer["📉 损失层"]
        C1[预测值 y_hat]
        C2[真实标签 y]
        C3[cross_entropy<br/>负对数似然]
        C1 --> C3
        C2 --> C3
    end

    subgraph Optimization_Layer["⚙️ 优化层"]
        D1[SGD 优化器]
        D2[学习率 lr=0.1]
        D3[梯度更新]
        D2 --> D1
        D1 --> D3
    end

    subgraph Training_Loop["🔄 训练循环"]
        E1[train_epoch_ch3]
        E2[evaluate_accuracy]
        E3[Accumulator<br/>指标累加]
        E1 --> E3
        E2 --> E3
    end

    subgraph Visualization_Layer["📊 可视化层"]
        F1[Animator 动画]
        F2[show_images<br/>图像展示]
        F3[predict_ch3<br/>预测展示]
    end

    subgraph Main_Flow["🚀 主流程"]
        G1[初始化参数 W, b]
        G2[加载数据]
        G3[训练 10 epochs]
        G4[评估准确率]
    end

    %% 数据流
    A2 --> B1
    B5 --> C1
    C3 --> D1
    D3 --> B2
    D3 --> B3
    
    %% 训练流
    G2 --> E1
    E1 --> E2
    E1 --> F1
    E2 --> F1
    G3 --> E1
    G4 --> E2
    
    %% 预测流
    E2 --> F3
    F3 --> F2

    style Data_Layer fill:#e3f2fd
    style Model_Layer fill:#e8f5e9
    style Loss_Layer fill:#fff3e0
    style Optimization_Layer fill:#fce4ec
    style Training_Loop fill:#f3e5f5
    style Visualization_Layer fill:#e0f7fa
    style Main_Flow fill:#fffde7
```


## Mermaid Map For *Concise Implementation*
```mermaid
graph TB
    subgraph Data_Layer["📦 数据层"]
        A1["synthetic_data<br/>合成数据生成"]
        A2["features: 1000×2"]
        A3["labels: 1000×1"]
        A4["TensorDataset"]
        A5["DataLoader<br/>batch_size=10"]
        A1 --> A2
        A1 --> A3
        A2 --> A4
        A3 --> A4
        A4 --> A5
    end

    subgraph Model_Layer["🧠 模型层"]
        B1["nn.Sequential"]
        B2["nn.Linear<br/>输入: 2, 输出: 1"]
        B3["权重 W<br/>2x1 初始化 N(0,0.01)"]
        B4["偏置 b<br/>1维 初始化为0"]
        B1 --> B2
        B2 --> B3
        B2 --> B4
    end

    subgraph Loss_Layer["📉 损失层"]
        C1["nn.MSELoss"]
        C2["均方误差<br/>L2 范数平方"]
        C3["预测值 y_hat = net(X)"]
        C4["真实值 y"]
        C3 --> C1
        C4 --> C1
        C1 --> C2
    end

    subgraph Optimization_Layer["⚙️ 优化层"]
        D1["torch.optim.SGD"]
        D2["学习率 lr=0.03"]
        D3["net.parameters"]
        D4["梯度更新"]
        D2 --> D1
        D3 --> D1
        D1 --> D4
    end

    subgraph Training_Loop["🔄 训练循环"]
        E1["epoch: 3轮"]
        E2["trainer.zero_grad"]
        E3["l.backward<br/>反向传播"]
        E4["trainer.step<br/>参数更新"]
        E5["计算总损失"]
        E1 --> E2
        E2 --> E3
        E3 --> E4
        E4 --> E5
    end

    subgraph Evaluation_Layer["📊 评估层"]
        F1["估计权重 w_hat"]
        F2["估计偏置 b_hat"]
        F3["真实权重 w=[2, -3.4]"]
        F4["真实偏置 b=4.2"]
        F5["参数误差分析"]
        F1 --> F5
        F2 --> F5
        F3 --> F5
        F4 --> F5
    end

    A5 --> B2
    B2 --> C3
    C1 --> D1
    D4 --> B3
    D4 --> B4
    E5 --> F1
    E5 --> F2

    style Data_Layer fill:#e3f2fd,stroke:#1976d2
    style Model_Layer fill:#e8f5e9,stroke:#388e3c
    style Loss_Layer fill:#fff3e0,stroke:#f57c00
    style Optimization_Layer fill:#fce4ec,stroke:#c2185b
    style Training_Loop fill:#f3e5f5,stroke:#7b1fa2
    style Evaluation_Layer fill:#e0f7fa,stroke:#00838f
```





