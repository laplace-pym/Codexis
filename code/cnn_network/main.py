"""
完整的CNN网络实现
包含CNN模型定义、训练和测试功能
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
import warnings
warnings.filterwarnings('ignore')

# 设置随机种子以确保可重复性
torch.manual_seed(42)
np.random.seed(42)

class SimpleCNN(nn.Module):
    """简单的CNN网络用于图像分类"""
    
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        
        # 卷积层
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        # 池化层
        self.pool = nn.MaxPool2d(2, 2)
        
        # 全连接层
        self.fc1 = nn.Linear(128 * 3 * 3, 256)  # 假设输入是28x28，经过3次池化后为3x3
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)
        
        # Dropout用于防止过拟合
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        # 卷积层 + ReLU激活 + 池化
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        
        # 展平
        x = x.view(-1, 128 * 3 * 3)
        
        # 全连接层
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x

class AdvancedCNN(nn.Module):
    """更高级的CNN网络，包含批量归一化和更多层"""
    
    def __init__(self, num_classes=10):
        super(AdvancedCNN, self).__init__()
        
        # 卷积块1
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        
        # 卷积块2
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        
        # 卷积块3
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        
        # 全连接层
        self.fc_layers = nn.Sequential(
            nn.Linear(128 * 4 * 4, 512),  # 假设输入是32x32，经过3次池化后为4x4
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        
        # 展平
        x = x.view(x.size(0), -1)
        
        # 全连接层
        x = self.fc_layers(x)
        
        return x

class CNNClassifier:
    """CNN分类器包装类，包含训练和评估功能"""
    
    def __init__(self, model_type='simple', num_classes=10, device=None):
        """
        初始化CNN分类器
        
        Args:
            model_type: 'simple' 或 'advanced'
            num_classes: 分类数量
            device: 计算设备 (cuda/cpu)
        """
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
            
        if model_type == 'simple':
            self.model = SimpleCNN(num_classes=num_classes)
        elif model_type == 'advanced':
            self.model = AdvancedCNN(num_classes=num_classes)
        else:
            raise ValueError(f"未知的模型类型: {model_type}")
            
        self.model = self.model.to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        
    def train(self, train_loader, val_loader=None, epochs=10, lr=0.001):
        """训练模型"""
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
        
        train_losses = []
        val_losses = []
        train_accuracies = []
        val_accuracies = []
        
        print(f"开始训练，使用设备: {self.device}")
        print(f"训练集大小: {len(train_loader.dataset)}")
        
        for epoch in range(epochs):
            # 训练阶段
            self.model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(self.device), target.to(self.device)
                
                optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
                
                if batch_idx % 100 == 0:
                    print(f'Epoch: {epoch+1}/{epochs} | Batch: {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f}')
            
            train_loss = running_loss / len(train_loader)
            train_accuracy = 100. * correct / total
            train_losses.append(train_loss)
            train_accuracies.append(train_accuracy)
            
            # 验证阶段
            if val_loader is not None:
                val_loss, val_accuracy = self.evaluate(val_loader)
                val_losses.append(val_loss)
                val_accuracies.append(val_accuracy)
                
                print(f'Epoch: {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Train Acc: {train_accuracy:.2f}% | '
                      f'Val Loss: {val_loss:.4f} | Val Acc: {val_accuracy:.2f}%')
            else:
                print(f'Epoch: {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Train Acc: {train_accuracy:.2f}%')
            
            scheduler.step()
        
        return {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'train_accuracies': train_accuracies,
            'val_accuracies': val_accuracies
        }
    
    def evaluate(self, data_loader):
        """评估模型"""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in data_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                
                running_loss += loss.item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
        
        avg_loss = running_loss / len(data_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def predict(self, data):
        """预测"""
        self.model.eval()
        with torch.no_grad():
            data = data.to(self.device)
            output = self.model(data)
            _, predicted = output.max(1)
        return predicted.cpu()
    
    def save_model(self, path):
        """保存模型"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'model_type': type(self.model).__name__
        }, path)
        print(f"模型已保存到: {path}")
    
    def load_model(self, path):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"模型已从 {path} 加载")

def create_synthetic_data(num_samples=1000, image_size=28, num_classes=10):
    """创建合成图像数据用于演示"""
    # 创建随机图像数据
    images = np.random.randn(num_samples, 1, image_size, image_size).astype(np.float32)
    labels = np.random.randint(0, num_classes, size=num_samples)
    
    # 转换为PyTorch张量
    images_tensor = torch.from_numpy(images)
    labels_tensor = torch.from_numpy(labels).long()
    
    return images_tensor, labels_tensor

def plot_training_history(history):
    """绘制训练历史"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # 损失曲线
    axes[0].plot(history['train_losses'], label='Train Loss')
    if history['val_losses']:
        axes[0].plot(history['val_losses'], label='Val Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    # 准确率曲线
    axes[1].plot(history['train_accuracies'], label='Train Accuracy')
    if history['val_accuracies']:
        axes[1].plot(history['val_accuracies'], label='Val Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('Training and Validation Accuracy')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.show()

def main():
    """主函数：演示CNN网络的完整流程"""
    print("=" * 60)
    print("CNN网络演示")
    print("=" * 60)
    
    # 1. 创建合成数据
    print("\n1. 创建合成数据...")
    images, labels = create_synthetic_data(num_samples=1000, image_size=28, num_classes=10)
    
    # 2. 划分训练集和测试集
    print("2. 划分数据集...")
    X_train, X_test, y_train, y_test = train_test_split(
        images, labels, test_size=0.2, random_state=42
    )
    
    # 3. 创建数据加载器
    print("3. 创建数据加载器...")
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # 4. 创建和训练模型
    print("4. 创建和训练CNN模型...")
    classifier = CNNClassifier(model_type='simple', num_classes=10)
    
    # 打印模型结构
    print("\n模型结构:")
    print(classifier.model)
    
    # 计算模型参数数量
    total_params = sum(p.numel() for p in classifier.model.parameters())
    trainable_params = sum(p.numel() for p in classifier.model.parameters() if p.requires_grad)
    print(f"\n总参数数量: {total_params:,}")
    print(f"可训练参数数量: {trainable_params:,}")
    
    # 训练模型
    print("\n开始训练...")
    history = classifier.train(
        train_loader=train_loader,
        val_loader=test_loader,
        epochs=5,
        lr=0.001
    )
    
    # 5. 评估模型
    print("\n5. 评估模型...")
    test_loss, test_accuracy = classifier.evaluate(test_loader)
    print(f"测试集损失: {test_loss:.4f}")
    print(f"测试集准确率: {test_accuracy:.2f}%")
    
    # 6. 绘制训练历史
    print("\n6. 绘制训练历史...")
    plot_training_history(history)
    
    # 7. 保存模型
    print("\n7. 保存模型...")
    classifier.save_model('./code/cnn_network/cnn_model.pth')
    
    # 8. 演示预测
    print("\n8. 演示预测...")
    # 使用测试集的前5个样本进行预测
    sample_data, sample_labels = next(iter(test_loader))
    predictions = classifier.predict(sample_data[:5])
    
    print(f"真实标签: {sample_labels[:5].numpy()}")
    print(f"预测标签: {predictions.numpy()}")
    
    print("\n" + "=" * 60)
    print("CNN网络演示完成!")
    print("=" * 60)

def advanced_cnn_demo():
    """高级CNN演示"""
    print("\n" + "=" * 60)
    print("高级CNN网络演示")
    print("=" * 60)
    
    # 创建彩色图像数据 (3通道)
    images = np.random.randn(500, 3, 32, 32).astype(np.float32)
    labels = np.random.randint(0, 5, size=500)
    
    images_tensor = torch.from_numpy(images)
    labels_tensor = torch.from_numpy(labels).long()
    
    # 创建数据加载器
    dataset = TensorDataset(images_tensor, labels_tensor)
    loader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    # 创建高级CNN模型
    print("\n创建高级CNN模型...")
    classifier = CNNClassifier(model_type='advanced', num_classes=5)
    
    # 打印模型结构
    print("\n高级CNN模型结构:")
    print(classifier.model)
    
    # 计算参数数量
    total_params = sum(p.numel() for p in classifier.model.parameters())
    trainable_params = sum(p.numel() for p in classifier.model.parameters() if p.requires_grad)
    print(f"\n总参数数量: {total_params:,}")
    print(f"可训练参数数量: {trainable_params:,}")
    
    # 快速训练几轮
    print("\n快速训练...")
    history = classifier.train(
        train_loader=loader,
        epochs=3,
        lr=0.001
    )
    
    print("\n高级CNN演示完成!")

if __name__ == "__main__":
    # 运行主演示
    main()
    
    # 运行高级CNN演示
    advanced_cnn_demo()