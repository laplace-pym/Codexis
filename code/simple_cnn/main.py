import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# 定义一个最简单的CNN模型
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # 第一个卷积层：输入通道1，输出通道16，卷积核3x3
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        # 第二个卷积层：输入通道16，输出通道32，卷积核3x3
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        # 最大池化层：2x2窗口
        self.pool = nn.MaxPool2d(2, 2)
        # 全连接层
        self.fc1 = nn.Linear(32 * 7 * 7, 128)  # 经过两次池化后，28x28 -> 14x14 -> 7x7
        self.fc2 = nn.Linear(128, 10)  # 10个类别（MNIST数字0-9）
        
    def forward(self, x):
        # 卷积 -> ReLU -> 池化
        x = self.pool(F.relu(self.conv1(x)))
        # 卷积 -> ReLU -> 池化
        x = self.pool(F.relu(self.conv2(x)))
        # 展平
        x = x.view(-1, 32 * 7 * 7)
        # 全连接层 -> ReLU
        x = F.relu(self.fc1(x))
        # 输出层
        x = self.fc2(x)
        return x

def train_model(model, device, train_loader, optimizer, epoch):
    """训练模型"""
    model.train()
    train_loss = 0
    correct = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        # 梯度清零
        optimizer.zero_grad()
        
        # 前向传播
        output = model(data)
        
        # 计算损失
        loss = F.cross_entropy(output, target)
        
        # 反向传播
        loss.backward()
        
        # 更新参数
        optimizer.step()
        
        # 统计
        train_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        
        # 每100个batch打印一次进度
        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')
    
    # 计算平均损失和准确率
    avg_loss = train_loss / len(train_loader)
    accuracy = 100. * correct / len(train_loader.dataset)
    print(f'\nTraining set: Average loss: {avg_loss:.4f}, Accuracy: {correct}/{len(train_loader.dataset)} ({accuracy:.2f}%)\n')
    
    return avg_loss, accuracy

def test_model(model, device, test_loader):
    """测试模型"""
    model.eval()
    test_loss = 0
    correct = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    
    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    
    print(f'Test set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)\n')
    
    return test_loss, accuracy

def visualize_predictions(model, device, test_loader, num_images=5):
    """可视化预测结果"""
    model.eval()
    data_iter = iter(test_loader)
    images, labels = next(data_iter)
    
    # 获取预测结果
    with torch.no_grad():
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
    
    # 显示图像和预测结果
    fig, axes = plt.subplots(1, num_images, figsize=(12, 3))
    for i in range(num_images):
        axes[i].imshow(images[i].cpu().squeeze(), cmap='gray')
        axes[i].set_title(f'True: {labels[i].item()}, Pred: {predicted[i].item()}')
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()

def main():
    # 设置随机种子以保证可重复性
    torch.manual_seed(42)
    
    # 检查是否有GPU可用
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 数据预处理
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST的均值和标准差
    ])
    
    # 加载MNIST数据集
    print("Loading MNIST dataset...")
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)
    
    # 创建数据加载器
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000, shuffle=False)
    
    # 创建模型
    model = SimpleCNN().to(device)
    print("Model architecture:")
    print(model)
    
    # 计算模型参数数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # 定义优化器
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 训练模型
    print("\nStarting training...")
    num_epochs = 5
    train_losses = []
    train_accuracies = []
    test_losses = []
    test_accuracies = []
    
    for epoch in range(1, num_epochs + 1):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch}/{num_epochs}")
        print(f"{'='*50}")
        
        # 训练
        train_loss, train_acc = train_model(model, device, train_loader, optimizer, epoch)
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        
        # 测试
        test_loss, test_acc = test_model(model, device, test_loader)
        test_losses.append(test_loss)
        test_accuracies.append(test_acc)
    
    # 可视化训练过程
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # 损失曲线
    ax1.plot(range(1, num_epochs + 1), train_losses, 'b-', label='Training Loss')
    ax1.plot(range(1, num_epochs + 1), test_losses, 'r-', label='Test Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Test Loss')
    ax1.legend()
    ax1.grid(True)
    
    # 准确率曲线
    ax2.plot(range(1, num_epochs + 1), train_accuracies, 'b-', label='Training Accuracy')
    ax2.plot(range(1, num_epochs + 1), test_accuracies, 'r-', label='Test Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training and Test Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # 可视化一些预测结果
    print("\nVisualizing predictions...")
    visualize_predictions(model, device, test_loader)
    
    # 保存模型
    torch.save(model.state_dict(), 'simple_cnn_mnist.pth')
    print("\nModel saved as 'simple_cnn_mnist.pth'")

if __name__ == "__main__":
    main()