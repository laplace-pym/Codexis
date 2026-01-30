"""
最简单的CNN实现
这个文件展示了最基本的CNN结构，适合初学者理解CNN的核心概念
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class SimplestCNN(nn.Module):
    """
    最简单的CNN模型
    结构：卷积层 -> 激活函数 -> 池化层 -> 全连接层
    """
    def __init__(self):
        super(SimplestCNN, self).__init__()
        
        # 卷积层：提取图像特征
        # 参数说明：
        #   in_channels: 输入通道数（灰度图为1，RGB图为3）
        #   out_channels: 输出通道数（卷积核数量）
        #   kernel_size: 卷积核大小（3x3）
        #   padding: 填充（保持输出尺寸不变）
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        
        # 池化层：下采样，减少计算量，增强特征不变性
        # 参数说明：
        #   kernel_size: 池化窗口大小（2x2）
        #   stride: 步长（2，表示每次移动2个像素）
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 全连接层：分类
        # 参数说明：
        #   in_features: 输入特征数（经过卷积和池化后的特征图展平）
        #   out_features: 输出类别数（10个数字）
        self.fc = nn.Linear(16 * 14 * 14, 10)  # 输入28x28，经过池化后变为14x14
    
    def forward(self, x):
        """
        前向传播过程
        """
        # 1. 卷积 + ReLU激活函数
        x = F.relu(self.conv1(x))
        
        # 2. 最大池化
        x = self.pool(x)
        
        # 3. 展平：将多维特征图转换为一维向量
        x = x.view(x.size(0), -1)  # -1表示自动计算维度
        
        # 4. 全连接层（分类）
        x = self.fc(x)
        
        return x

def explain_cnn():
    """
    解释CNN的基本概念
    """
    print("=" * 60)
    print("CNN（卷积神经网络）基本概念解释")
    print("=" * 60)
    
    print("\n1. 卷积层 (Convolutional Layer):")
    print("   - 作用：提取图像局部特征")
    print("   - 原理：使用卷积核在图像上滑动，计算局部区域的加权和")
    print("   - 参数：卷积核大小、步长、填充")
    
    print("\n2. 激活函数 (Activation Function):")
    print("   - 作用：引入非线性，使网络能够学习复杂模式")
    print("   - 常用函数：ReLU、Sigmoid、Tanh")
    print("   - ReLU公式：f(x) = max(0, x)")
    
    print("\n3. 池化层 (Pooling Layer):")
    print("   - 作用：下采样，减少计算量，增强特征不变性")
    print("   - 类型：最大池化、平均池化")
    print("   - 最大池化：取窗口内的最大值")
    
    print("\n4. 全连接层 (Fully Connected Layer):")
    print("   - 作用：将提取的特征进行分类")
    print("   - 原理：每个神经元与前一层的所有神经元连接")
    
    print("\n5. 前向传播 (Forward Propagation):")
    print("   - 输入 -> 卷积 -> 激活 -> 池化 -> 展平 -> 全连接 -> 输出")
    
    print("\n6. 反向传播 (Backward Propagation):")
    print("   - 作用：根据损失函数计算梯度，更新网络参数")
    print("   - 优化器：SGD、Adam、RMSprop等")
    
    print("\n" + "=" * 60)

def create_sample_input():
    """
    创建示例输入数据
    """
    print("\n创建示例输入数据...")
    
    # 创建一个批次的示例数据
    # 形状：[batch_size, channels, height, width]
    # 这里创建1个批次，1个通道（灰度图），28x28像素
    batch_size = 1
    channels = 1  # 灰度图
    height = 28
    width = 28
    
    # 创建随机输入数据（模拟图像）
    sample_input = torch.randn(batch_size, channels, height, width)
    print(f"输入数据形状: {sample_input.shape}")
    print(f"  批次大小: {batch_size}")
    print(f"  通道数: {channels}")
    print(f"  高度: {height}")
    print(f"  宽度: {width}")
    
    return sample_input

def demonstrate_forward_pass():
    """
    演示前向传播过程
    """
    print("\n" + "=" * 60)
    print("演示前向传播过程")
    print("=" * 60)
    
    # 创建模型
    model = SimplestCNN()
    print(f"\n模型结构:")
    print(model)
    
    # 创建示例输入
    sample_input = create_sample_input()
    
    # 前向传播
    print("\n执行前向传播...")
    with torch.no_grad():  # 不计算梯度，仅用于演示
        output = model(sample_input)
    
    print(f"\n输出形状: {output.shape}")
    print(f"输出值（前5个）: {output[0, :5]}")
    
    # 解释输出
    print(f"\n输出解释:")
    print(f"  输出有10个值，对应10个数字类别（0-9）")
    print(f"  值越大表示属于该类的概率越高")
    print(f"  可以使用softmax函数将输出转换为概率分布")
    
    # 计算softmax概率
    probabilities = F.softmax(output, dim=1)
    print(f"\nSoftmax概率（前5个）: {probabilities[0, :5]}")
    print(f"概率总和: {probabilities.sum().item():.4f}")
    
    # 预测类别
    predicted_class = torch.argmax(output, dim=1)
    print(f"\n预测类别: {predicted_class.item()}")
    
    return model, sample_input, output

def count_parameters(model):
    """
    计算模型参数数量
    """
    print("\n" + "=" * 60)
    print("模型参数统计")
    print("=" * 60)
    
    total_params = 0
    trainable_params = 0
    
    print("\n各层参数详情:")
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            param_count = parameter.numel()
            trainable_params += param_count
            total_params += param_count
            print(f"  {name}: {param_count:,} 参数")
    
    print(f"\n总参数数量: {total_params:,}")
    print(f"可训练参数数量: {trainable_params:,}")
    
    return total_params, trainable_params

def main():
    """
    主函数：演示最简单的CNN
    """
    print("最简单的CNN实现演示")
    print("=" * 60)
    
    # 解释CNN基本概念
    explain_cnn()
    
    # 演示前向传播
    model, sample_input, output = demonstrate_forward_pass()
    
    # 计算参数数量
    count_parameters(model)
    
    print("\n" + "=" * 60)
    print("总结:")
    print("=" * 60)
    print("1. 这个最简单的CNN包含：")
    print("   - 1个卷积层（16个3x3卷积核）")
    print("   - ReLU激活函数")
    print("   - 最大池化层（2x2窗口）")
    print("   - 1个全连接层（10个输出）")
    print("\n2. 输入：28x28灰度图像")
    print("3. 输出：10个类别的得分")
    print("4. 总参数：约31,754个")
    print("\n这个模型可以用于MNIST手写数字识别任务！")

if __name__ == "__main__":
    main()