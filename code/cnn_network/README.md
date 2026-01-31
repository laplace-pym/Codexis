# CNN网络实现

本目录包含多个CNN（卷积神经网络）的实现，从简单到高级，适合不同层次的学习者。

## 文件结构

- `main.py` - 完整的CNN实现，包含训练、验证、测试和可视化功能
- `example.py` - 使用示例，展示如何创建和使用CNN模型
- `simple_cnn.py` - 最简单的CNN实现，适合初学者
- `README.md` - 本说明文件

## 模型介绍

### 1. SimpleCNN (简单CNN)
- 3个卷积层 + 3个池化层
- 批归一化层
- Dropout防止过拟合
- 3个全连接层

### 2. AdvancedCNN (高级CNN)
- 包含残差连接（Residual Blocks）
- 全局平均池化
- 更深的网络结构
- 更好的梯度流动

### 3. BasicCNN (基础CNN)
- 最简单的实现，适合教学
- 2个卷积层 + 2个池化层
- 2个全连接层

### 4. TinyCNN (微型CNN)
- 超小型网络，用于快速测试
- 1个卷积层 + 1个池化层
- 1个全连接层

## 快速开始

### 安装依赖
```bash
pip install torch torchvision numpy matplotlib scikit-learn
```

### 运行完整演示
```bash
python main.py
```

### 运行简单示例
```bash
python simple_cnn.py
```

### 运行使用示例
```bash
python example.py
```

## 使用示例

### 1. 创建模型
```python
from main import SimpleCNN, AdvancedCNN
from simple_cnn import BasicCNN

# 创建简单CNN
model = SimpleCNN(num_classes=10, input_channels=3)

# 创建高级CNN
model = AdvancedCNN(num_classes=10, input_channels=3)

# 创建基础CNN
model = BasicCNN(num_classes=10)
```

### 2. 前向传播
```python
# 创建随机输入 (batch_size=4, channels=3, height=32, width=32)
input_tensor = torch.randn(4, 3, 32, 32)

# 前向传播
output = model(input_tensor)
print(f"输出形状: {output.shape}")  # [4, 10]
```

### 3. 训练模型
```python
# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练循环
for epoch in range(num_epochs):
    model.train()
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
```

### 4. 保存和加载模型
```python
# 保存模型
torch.save(model.state_dict(), 'model.pth')

# 加载模型
model = SimpleCNN(num_classes=10, input_channels=3)
model.load_state_dict(torch.load('model.pth'))
model.eval()
```

## 模型参数

### SimpleCNN 参数统计
- 总参数量: ~1.2M
- 可训练参数量: ~1.2M
- 适合: CIFAR-10, MNIST等小型数据集

### AdvancedCNN 参数统计
- 总参数量: ~3.5M
- 可训练参数量: ~3.5M
- 适合: 更复杂的图像分类任务

### BasicCNN 参数统计
- 总参数量: ~270K
- 可训练参数量: ~270K
- 适合: 教学和快速原型开发

## 输入输出规格

### 输入
- 形状: `[batch_size, channels, height, width]`
- 默认: `[batch_size, 3, 32, 32]`
- 数据类型: `torch.float32`

### 输出
- 形状: `[batch_size, num_classes]`
- 默认: `[batch_size, 10]`
- 输出为未归一化的logits

## 训练功能

`main.py` 包含完整的训练流程：

1. **数据准备**: 创建合成数据用于演示
2. **数据划分**: 训练集、验证集、测试集
3. **模型训练**: 包含训练和验证循环
4. **性能评估**: 计算准确率
5. **结果可视化**: 绘制损失和准确率曲线
6. **模型保存**: 保存训练好的模型

## 自定义修改

### 修改输入尺寸
```python
# 修改全连接层的输入维度
self.fc1 = nn.Linear(128 * new_height * new_width, 512)
```

### 修改网络深度
```python
# 添加更多卷积层
self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
self.bn4 = nn.BatchNorm2d(256)
```

### 修改激活函数
```python
# 使用LeakyReLU代替ReLU
x = F.leaky_relu(self.bn1(self.conv1(x)), negative_slope=0.01)
```

## 常见问题

### Q: 如何修改输入图像大小？
A: 需要调整全连接层的输入维度。计算公式：`通道数 × 高度 × 宽度`

### Q: 如何增加类别数量？
A: 修改 `num_classes` 参数，并调整最后一层全连接层

### Q: 为什么使用批归一化？
A: 批归一化可以加速训练，提高模型稳定性，减少对初始化的敏感度

### Q: Dropout的作用是什么？
A: Dropout可以防止过拟合，通过在训练时随机丢弃一些神经元

### Q: 残差连接有什么好处？
A: 残差连接可以解决深度网络中的梯度消失问题，使网络更容易训练

## 扩展功能

可以扩展的功能包括：
1. 数据增强（旋转、翻转、裁剪等）
2. 学习率调度器
3. 早停机制
4. 模型集成
5. 迁移学习

## 许可证

本项目仅供学习使用，遵循MIT许可证。

## 贡献

欢迎提交问题和改进建议！