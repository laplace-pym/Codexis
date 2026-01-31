"""
最简单的CNN实现
适合初学者学习
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicCNN(nn.Module):
    """
    最基本的CNN网络
    包含：卷积层、池化层、全连接层
    """
    
    def __init__(self, num_classes=10):
        super(BasicCNN, self).__init__()
        
        # 卷积层1: 输入3通道，输出16通道，3x3卷积核
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        
        # 卷积层2: 输入16通道，输出32通道，3x3卷积核
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        
        # 最大池化层: 2x2窗口，步长2
        self.pool = nn.MaxPool2d(2, 2)
        
        # 全连接层1: 假设输入图像32x32，经过2次池化后为8x8
        # 32通道 * 8 * 8 = 2048个特征
        self.fc1 = nn.Linear(32 * 8 * 8, 128)
        
        # 全连接层2: 输出层
        self.fc2 = nn.Linear(128, num_classes)
        
        # Dropout防止过拟合
        self.dropout = nn.Dropout(0.25)
    
    def forward(self, x):
        """
        前向传播过程
        x: 输入图像张量，形状为 [batch_size, 3, 32, 32]
        """
        # 第一卷积块
        x = self.pool(F.relu(self.conv1(x)))  # 输出: [batch_size, 16, 16, 16]
        
        # 第二卷积块
        x = self.pool(F.relu(self.conv2(x)))  # 输出: [batch_size, 32, 8, 8]
        
        # 展平特征图
        x = x.view(-1, 32 * 8 * 8)  # 形状: [batch_size, 2048]
        
        # 全连接层
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

class TinyCNN(nn.Module):
    """
    超小型CNN，用于快速测试
    """
    
    def __init__(self, num_classes=10):
        super(TinyCNN, self).__init__()
        
        # 单个卷积层
        self.conv = nn.Conv2d(3, 8, kernel_size=3, padding=1)
        
        # 池化层
        self.pool = nn.MaxPool2d(2, 2)
        
        # 全连接层
        self.fc = nn.Linear(8 * 16 * 16, num_classes)  # 32x32 -> 16x16 after pooling
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv(x)))
        x = x.view(-1, 8 * 16 * 16)
        x = self.fc(x)
        return x

def create_model(model_type='basic'):
    """
    创建CNN模型
    """
    if model_type == 'tiny':
        return TinyCNN()
    else:
        return BasicCNN()

def test_model():
    """测试模型"""
    print("测试CNN模型")
    print("=" * 40)
    
    # 创建模型
    model = BasicCNN(num_classes=10)
    print("模型创建成功!")
    print(f"模型结构:\n{model}")
    
    # 创建随机输入数据
    batch_size = 4
    dummy_input = torch.randn(batch_size, 3, 32, 32)
    print(f"\n输入数据形状: {dummy_input.shape}")
    
    # 前向传播
    output = model(dummy_input)
    print(f"输出形状: {output.shape}")
    
    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n模型参数量:")
    print(f"  总参数量: {total_params:,}")
    print(f"  可训练参数量: {trainable_params:,}")
    
    # 各层参数统计
    print(f"\n各层参数详情:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"  {name}: {param.shape} ({param.numel():,} 参数)")
    
    return model

def quick_start():
    """快速开始指南"""
    print("CNN快速开始指南")
    print("=" * 40)
    
    print("\n1. 导入必要的库:")
    print("""
import torch
import torch.nn as nn
import torch.nn.functional as F
""")
    
    print("\n2. 定义CNN模型:")
    print("""
class MyCNN(nn.Module):
    def __init__(self):
        super(MyCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 10)
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * 8 * 8)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
""")
    
    print("\n3. 创建模型实例:")
    print("""
model = MyCNN()
""")
    
    print("\n4. 准备输入数据:")
    print("""
# 创建随机输入 (batch_size=4, channels=3, height=32, width=32)
input_data = torch.randn(4, 3, 32, 32)
""")
    
    print("\n5. 前向传播:")
    print("""
output = model(input_data)
print(f"输出形状: {output.shape}")
""")
    
    print("\n6. 训练模型:")
    print("""
# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 训练循环
for epoch in range(10):
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
""")

if __name__ == "__main__":
    print("CNN网络实现")
    print("=" * 40)
    
    # 测试模型
    model = test_model()
    
    print("\n" + "=" * 40)
    print("快速开始指南:")
    quick_start()
    
    print("\n" + "=" * 40)
    print("完成! 您已经成功创建了一个CNN网络。")