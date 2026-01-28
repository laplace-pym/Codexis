import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from transformer import Transformer


class SimpleTranslationDataset(Dataset):
    """简单的翻译数据集（用于演示）"""
    
    def __init__(self, num_samples=1000, src_len=10, tgt_len=12, vocab_size=100):
        self.num_samples = num_samples
        self.src_len = src_len
        self.tgt_len = tgt_len
        self.vocab_size = vocab_size
        
        # 生成随机数据
        self.src_data = []
        self.tgt_data = []
        
        for _ in range(num_samples):
            # 源序列（去掉开始和结束标记的位置）
            src = torch.randint(3, vocab_size, (src_len,))
            src[0] = 1  # 开始标记
            src[-1] = 2  # 结束标记
            
            # 目标序列（比源序列稍长）
            tgt = torch.randint(3, vocab_size, (tgt_len,))
            tgt[0] = 1  # 开始标记
            tgt[-1] = 2  # 结束标记
            
            self.src_data.append(src)
            self.tgt_data.append(tgt)
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return self.src_data[idx], self.tgt_data[idx]


def collate_fn(batch):
    """批处理函数"""
    src_batch, tgt_batch = zip(*batch)
    
    # 填充到相同长度
    src_len = max(len(src) for src in src_batch)
    tgt_len = max(len(tgt) for tgt in tgt_batch)
    
    src_padded = torch.zeros(len(batch), src_len, dtype=torch.long)
    tgt_padded = torch.zeros(len(batch), tgt_len, dtype=torch.long)
    
    for i, (src, tgt) in enumerate(batch):
        src_padded[i, :len(src)] = src
        tgt_padded[i, :len(tgt)] = tgt
    
    return src_padded, tgt_padded


def train_transformer():
    """训练Transformer模型"""
    print("开始训练Transformer模型...")
    
    # 超参数
    src_vocab_size = 100
    tgt_vocab_size = 100
    d_model = 128
    num_layers = 4
    num_heads = 8
    d_ff = 512
    batch_size = 32
    num_epochs = 20
    learning_rate = 0.0005
    
    # 创建数据集和数据加载器
    dataset = SimpleTranslationDataset(num_samples=1000)
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True,
        collate_fn=collate_fn
    )
    
    # 创建模型
    model = Transformer(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        d_ff=d_ff,
        dropout=0.1
    )
    
    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # 忽略padding
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    
    # 训练循环
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        for batch_idx, (src, tgt) in enumerate(dataloader):
            # 前向传播
            output = model(src, tgt[:, :-1])  # 使用tgt[:-1]作为输入
            
            # 计算损失
            loss = criterion(
                output.reshape(-1, tgt_vocab_size),
                tgt[:, 1:].reshape(-1)  # 使用tgt[1:]作为目标
            )
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            
            # 梯度裁剪（防止梯度爆炸）
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 10 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], "
                      f"Batch [{batch_idx}/{len(dataloader)}], "
                      f"Loss: {loss.item():.4f}")
        
        # 更新学习率
        scheduler.step()
        
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{num_epochs}], "
              f"Average Loss: {avg_loss:.4f}, "
              f"LR: {scheduler.get_last_lr()[0]:.6f}")
    
    print("训练完成！")
    return model


def evaluate_model(model, num_examples=5):
    """评估模型"""
    print("\n评估模型...")
    model.eval()
    
    # 创建测试数据
    dataset = SimpleTranslationDataset(num_samples=num_examples)
    
    with torch.no_grad():
        for i in range(num_examples):
            src, tgt = dataset[i]
            
            # 添加批次维度
            src = src.unsqueeze(0)
            
            print(f"\n示例 {i+1}:")
            print(f"源序列: {src[0].tolist()}")
            print(f"目标序列: {tgt.tolist()}")
            
            # 生成序列
            generated = model.generate(src, max_len=15)
            print(f"生成序列: {generated[0].tolist()}")
            
            # 计算准确率（仅比较非padding部分）
            pred_tokens = generated[0].tolist()
            true_tokens = tgt.tolist()
            
            # 找到结束标记
            try:
                pred_end = pred_tokens.index(2) if 2 in pred_tokens else len(pred_tokens)
                true_end = true_tokens.index(2) if 2 in true_tokens else len(true_tokens)
                
                # 比较序列
                min_len = min(pred_end, true_end)
                if min_len > 1:  # 跳过开始标记
                    matches = sum(1 for j in range(1, min_len) 
                                 if pred_tokens[j] == true_tokens[j])
                    accuracy = matches / (min_len - 1)
                    print(f"准确率: {accuracy:.2%}")
            except:
                print("无法计算准确率")


def save_and_load_model(model):
    """保存和加载模型"""
    print("\n保存和加载模型...")
    
    # 保存模型
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': {
            'src_vocab_size': 100,
            'tgt_vocab_size': 100,
            'd_model': 128,
            'num_layers': 4,
            'num_heads': 8,
            'd_ff': 512,
            'dropout': 0.1
        }
    }, 'transformer_model.pth')
    
    print("模型已保存到 transformer_model.pth")
    
    # 加载模型
    checkpoint = torch.load('transformer_model.pth')
    
    # 创建新模型
    loaded_model = Transformer(**checkpoint['config'])
    loaded_model.load_state_dict(checkpoint['model_state_dict'])
    
    print("模型已加载")
    
    return loaded_model


def main():
    """主函数"""
    print("Transformer训练示例")
    print("="*60)
    
    # 训练模型
    model = train_transformer()
    
    # 评估模型
    evaluate_model(model)
    
    # 保存和加载模型
    loaded_model = save_and_load_model(model)
    
    # 测试加载的模型
    print("\n测试加载的模型...")
    evaluate_model(loaded_model, num_examples=2)
    
    print("\n" + "="*60)
    print("示例完成！")
    print("="*60)


if __name__ == "__main__":
    main()