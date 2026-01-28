"""
Transformer模型训练脚本
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
from transformer import Transformer


class DummyTranslationDataset(Dataset):
    """虚拟翻译数据集（用于演示）"""
    def __init__(self, num_samples=1000, src_vocab_size=100, tgt_vocab_size=100, max_len=20):
        self.num_samples = num_samples
        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        self.max_len = max_len
        
        # 生成虚拟数据
        self.data = []
        for _ in range(num_samples):
            # 随机生成句子长度
            src_len = random.randint(5, max_len)
            tgt_len = random.randint(5, max_len)
            
            # 生成句子（1是起始符，2是结束符，0是padding）
            src_sentence = [1] + [random.randint(3, src_vocab_size-1) for _ in range(src_len-2)] + [2]
            tgt_sentence = [1] + [random.randint(3, tgt_vocab_size-1) for _ in range(tgt_len-2)] + [2]
            
            # 填充到最大长度
            src_sentence = src_sentence + [0] * (max_len - len(src_sentence))
            tgt_sentence = tgt_sentence + [0] * (max_len - len(tgt_sentence))
            
            self.data.append((src_sentence, tgt_sentence))
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        src, tgt = self.data[idx]
        return torch.tensor(src, dtype=torch.long), torch.tensor(tgt, dtype=torch.long)


def train_epoch(model, dataloader, criterion, optimizer, device):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    total_correct = 0
    total_tokens = 0
    
    progress_bar = tqdm(dataloader, desc="训练")
    for src, tgt in progress_bar:
        src, tgt = src.to(device), tgt.to(device)
        
        # 准备输入输出
        tgt_input = tgt[:, :-1]  # 去掉最后一个token
        tgt_output = tgt[:, 1:]   # 去掉第一个token（shifted right）
        
        # 前向传播
        optimizer.zero_grad()
        output = model(src, tgt_input)
        
        # 计算损失
        loss = criterion(
            output.reshape(-1, output.size(-1)),
            tgt_output.reshape(-1)
        )
        
        # 反向传播
        loss.backward()
        
        # 梯度裁剪（防止梯度爆炸）
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # 优化器步进
        optimizer.step()
        
        # 统计
        total_loss += loss.item()
        
        # 计算准确率
        predictions = output.argmax(dim=-1)
        mask = (tgt_output != 0)  # 忽略padding
        correct = (predictions == tgt_output) & mask
        total_correct += correct.sum().item()
        total_tokens += mask.sum().item()
        
        # 更新进度条
        progress_bar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{correct.sum().item() / mask.sum().item():.4f}'
        })
    
    avg_loss = total_loss / len(dataloader)
    accuracy = total_correct / total_tokens if total_tokens > 0 else 0
    
    return avg_loss, accuracy


def evaluate(model, dataloader, criterion, device):
    """评估模型"""
    model.eval()
    total_loss = 0
    total_correct = 0
    total_tokens = 0
    
    with torch.no_grad():
        for src, tgt in tqdm(dataloader, desc="评估"):
            src, tgt = src.to(device), tgt.to(device)
            
            # 准备输入输出
            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]
            
            # 前向传播
            output = model(src, tgt_input)
            
            # 计算损失
            loss = criterion(
                output.reshape(-1, output.size(-1)),
                tgt_output.reshape(-1)
            )
            
            total_loss += loss.item()
            
            # 计算准确率
            predictions = output.argmax(dim=-1)
            mask = (tgt_output != 0)
            correct = (predictions == tgt_output) & mask
            total_correct += correct.sum().item()
            total_tokens += mask.sum().item()
    
    avg_loss = total_loss / len(dataloader)
    accuracy = total_correct / total_tokens if total_tokens > 0 else 0
    
    return avg_loss, accuracy


def plot_training_history(train_losses, val_losses, train_accs, val_accs):
    """绘制训练历史"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # 绘制损失
    epochs = range(1, len(train_losses) + 1)
    ax1.plot(epochs, train_losses, 'b-', label='训练损失')
    ax1.plot(epochs, val_losses, 'r-', label='验证损失')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('损失')
    ax1.set_title('训练和验证损失')
    ax1.legend()
    ax1.grid(True)
    
    # 绘制准确率
    ax2.plot(epochs, train_accs, 'b-', label='训练准确率')
    ax2.plot(epochs, val_accs, 'r-', label='验证准确率')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('准确率')
    ax2.set_title('训练和验证准确率')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('./code/transformer/training_history.png', dpi=150)
    plt.show()


def main():
    """主训练函数"""
    print("开始训练Transformer模型...")
    
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    # 设备设置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 数据集参数
    src_vocab_size = 100
    tgt_vocab_size = 100
    max_len = 20
    
    # 创建数据集
    train_dataset = DummyTranslationDataset(
        num_samples=800,
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        max_len=max_len
    )
    
    val_dataset = DummyTranslationDataset(
        num_samples=200,
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        max_len=max_len
    )
    
    # 创建数据加载器
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"训练集大小: {len(train_dataset)}")
    print(f"验证集大小: {len(val_dataset)}")
    
    # 创建模型
    model = Transformer(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        d_model=64,      # 较小的维度用于快速训练
        num_layers=2,    # 较少的层数
        num_heads=4,
        d_ff=256,
        dropout=0.1
    ).to(device)
    
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # 忽略padding
    optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.98), eps=1e-9)
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    
    # 训练参数
    num_epochs = 10
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    print(f"\n开始训练，共{num_epochs}个epoch...")
    
    # 训练循环
    for epoch in range(1, num_epochs + 1):
        print(f"\nEpoch {epoch}/{num_epochs}")
        print("-" * 50)
        
        # 训练
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # 评估
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        # 更新学习率
        scheduler.step()
        
        print(f"训练损失: {train_loss:.4f}, 训练准确率: {train_acc:.4f}")
        print(f"验证损失: {val_loss:.4f}, 验证准确率: {val_acc:.4f}")
        print(f"学习率: {scheduler.get_last_lr()[0]:.6f}")
        
        # 保存最佳模型
        if epoch == 1 or val_loss < min(val_losses[:-1]):
            torch.save(model.state_dict(), './code/transformer/best_model.pth')
            print(f"保存最佳模型 (epoch {epoch})")
    
    print("\n训练完成!")
    
    # 绘制训练历史
    try:
        plot_training_history(train_losses, val_losses, train_accs, val_accs)
        print("训练历史图已保存到: ./code/transformer/training_history.png")
    except Exception as e:
        print(f"绘制训练历史时出错: {e}")
    
    # 测试生成
    print("\n测试生成功能...")
    model.eval()
    
    # 使用验证集的一个样本
    src_sample, tgt_sample = val_dataset[0]
    src_sample = src_sample.unsqueeze(0).to(device)
    
    # 生成翻译
    with torch.no_grad():
        generated = model.generate(
            src_sample,
            max_len=15,
            start_token=1,
            end_token=2
        )
    
    print(f"源句子: {src_sample.cpu().numpy()[0]}")
    print(f"目标句子: {tgt_sample.numpy()}")
    print(f"生成句子: {generated.cpu().numpy()[0]}")
    
    # 保存最终模型
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accs': train_accs,
        'val_accs': val_accs,
        'epoch': num_epochs
    }, './code/transformer/final_model.pth')
    
    print("\n模型已保存到:")
    print("  - ./code/transformer/best_model.pth (最佳模型)")
    print("  - ./code/transformer/final_model.pth (最终模型+训练历史)")
    
    return model


if __name__ == "__main__":
    model = main()