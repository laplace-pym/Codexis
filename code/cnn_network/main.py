import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification

class SimpleCNN(nn.Module):
    """一个简单的CNN网络用于图像分类"""
    
    def __init__(self, num_classes=10, input_channels=3):
        super(SimpleCNN, self).__init__()
        
        # 卷积层
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        # 池化层
        self.pool = nn.MaxPool2d(2, 2)
        
        # Dropout层防止过拟合
        self.dropout = nn.Dropout(0.5)
        
        # 全连接层
        # 假设输入图像大小为32x32，经过3次池化后为4x4
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)
        
        # 批归一化层
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        
    def forward(self, x):
        # 第一卷积块
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        
        # 第二卷积块
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        
        # 第三卷积块
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        
        # 展平
        x = x.view(-1, 128 * 4 * 4)
        
        # 全连接层
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x

class AdvancedCNN(nn.Module):
    """更高级的CNN网络，包含残差连接"""
    
    def __init__(self, num_classes=10, input_channels=3):
        super(AdvancedCNN, self).__init__()
        
        # 初始卷积层
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # 残差块
        self.res_block1 = ResidualBlock(64, 128, stride=1)
        self.res_block2 = ResidualBlock(128, 256, stride=2)
        self.res_block3 = ResidualBlock(256, 512, stride=2)
        
        # 全局平均池化
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # 全连接层
        self.fc = nn.Linear(512, num_classes)
        
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.res_block3(x)
        
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        return x

class ResidualBlock(nn.Module):
    """残差块"""
    
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                              stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                              stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # 下采样连接
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
            
    def forward(self, x):
        identity = x
        
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        if self.downsample is not None:
            identity = self.downsample(x)
            
        out += identity
        out = F.relu(out)
        
        return out

def create_synthetic_data(num_samples=1000, image_size=32, num_channels=3, num_classes=10):
    """创建合成图像数据用于测试"""
    # 创建随机图像数据
    images = np.random.randn(num_samples, num_channels, image_size, image_size).astype(np.float32)
    labels = np.random.randint(0, num_classes, num_samples)
    
    # 转换为PyTorch张量
    images_tensor = torch.from_numpy(images)
    labels_tensor = torch.from_numpy(labels).long()
    
    return images_tensor, labels_tensor

def train_model(model, train_loader, val_loader, num_epochs=10, learning_rate=0.001):
    """训练CNN模型"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    
    print(f"Training on {device}")
    print("-" * 50)
    
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
        train_loss = running_loss / len(train_loader)
        train_accuracy = 100. * correct / total
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        val_loss = val_loss / len(val_loader)
        val_accuracy = 100. * correct / total
        
        # 记录指标
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_accuracy)
        val_accuracies.append(val_accuracy)
        
        print(f'Epoch [{epoch+1}/{num_epochs}]')
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}%')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}%')
        print("-" * 50)
    
    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accuracies': train_accuracies,
        'val_accuracies': val_accuracies
    }

def visualize_results(results):
    """可视化训练结果"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # 损失曲线
    axes[0].plot(results['train_losses'], label='Train Loss')
    axes[0].plot(results['val_losses'], label='Val Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    # 准确率曲线
    axes[1].plot(results['train_accuracies'], label='Train Accuracy')
    axes[1].plot(results['val_accuracies'], label='Val Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('Training and Validation Accuracy')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig('./code/cnn_network/training_results.png')
    plt.show()

def test_model(model, test_loader):
    """测试模型性能"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    accuracy = 100. * correct / total
    print(f'Test Accuracy: {accuracy:.2f}%')
    return accuracy

def main():
    """主函数：演示CNN网络的完整流程"""
    print("=" * 60)
    print("CNN网络演示")
    print("=" * 60)
    
    # 1. 创建合成数据
    print("\n1. 创建合成数据...")
    images, labels = create_synthetic_data(
        num_samples=2000,
        image_size=32,
        num_channels=3,
        num_classes=10
    )
    
    # 2. 划分数据集
    print("2. 划分数据集...")
    X_train, X_temp, y_train, y_temp = train_test_split(
        images.numpy(), labels.numpy(), test_size=0.3, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42
    )
    
    # 转换为张量
    X_train = torch.from_numpy(X_train)
    y_train = torch.from_numpy(y_train).long()
    X_val = torch.from_numpy(X_val)
    y_val = torch.from_numpy(y_val).long()
    X_test = torch.from_numpy(X_test)
    y_test = torch.from_numpy(y_test).long()
    
    # 3. 创建数据加载器
    print("3. 创建数据加载器...")
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    test_dataset = TensorDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # 4. 创建模型
    print("4. 创建CNN模型...")
    print("\n选项:")
    print("1. 简单CNN")
    print("2. 高级CNN（带残差连接）")
    
    choice = input("\n请选择模型类型 (1或2): ").strip()
    
    if choice == '2':
        model = AdvancedCNN(num_classes=10, input_channels=3)
        print("创建了高级CNN模型（带残差连接）")
    else:
        model = SimpleCNN(num_classes=10, input_channels=3)
        print("创建了简单CNN模型")
    
    # 打印模型结构
    print("\n模型结构:")
    print(model)
    
    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")
    
    # 5. 训练模型
    print("\n5. 训练模型...")
    results = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=10,
        learning_rate=0.001
    )
    
    # 6. 测试模型
    print("\n6. 测试模型...")
    test_accuracy = test_model(model, test_loader)
    
    # 7. 可视化结果
    print("\n7. 可视化训练结果...")
    visualize_results(results)
    
    # 8. 保存模型
    print("\n8. 保存模型...")
    torch.save(model.state_dict(), './code/cnn_network/cnn_model.pth')
    print("模型已保存到: ./code/cnn_network/cnn_model.pth")
    
    print("\n" + "=" * 60)
    print("CNN网络演示完成！")
    print("=" * 60)

if __name__ == "__main__":
    main()