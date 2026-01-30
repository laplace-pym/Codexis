"""
最精简的CNN实现 - 仅核心代码
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# 最简单的CNN类
class MinimalCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # 卷积层：1个输入通道 -> 16个输出通道，3x3卷积核
        self.conv = nn.Conv2d(1, 16, 3, padding=1)
        # 池化层：2x2最大池化
        self.pool = nn.MaxPool2d(2)
        # 全连接层：分类层
        self.fc = nn.Linear(16 * 14 * 14, 10)
    
    def forward(self, x):
        # 卷积 -> ReLU -> 池化
        x = self.pool(F.relu(self.conv(x)))
        # 展平
        x = x.view(x.size(0), -1)
        # 分类
        x = self.fc(x)
        return x

# 使用示例
if __name__ == "__main__":
    # 创建模型
    model = MinimalCNN()
    print("模型结构:")
    print(model)
    
    # 创建随机输入（模拟1张28x28的灰度图）
    input_tensor = torch.randn(1, 1, 28, 28)
    print(f"\n输入形状: {input_tensor.shape}")
    
    # 前向传播
    output = model(input_tensor)
    print(f"输出形状: {output.shape}")
    print(f"输出值: {output}")
    
    # 计算参数数量
    params = sum(p.numel() for p in model.parameters())
    print(f"\n总参数数量: {params:,}")
    
    print("\n这就是最简单的CNN！")
    print("包含：1个卷积层 + ReLU + 池化层 + 1个全连接层")