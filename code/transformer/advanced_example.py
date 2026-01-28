"""
高级Transformer示例：机器翻译任务
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from transformer import Transformer


class TranslationDataset:
    """简单的翻译数据集"""
    def __init__(self, num_samples=1000, max_len=20, vocab_size=100):
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.num_samples = num_samples
        
        # 生成示例数据（模拟翻译任务）
        self.data = []
        for _ in range(num_samples):
            # 生成源序列（英语）
            src_len = np.random.randint(5, max_len-2)
            src = np.random.randint(1, vocab_size//2, src_len)
            src = np.pad(src, (0, max_len - src_len), 'constant')
            
            # 生成目标序列（法语）- 模拟翻译
            tgt_len = np.random.randint(5, max_len-2)
            tgt = src[:tgt_len] + vocab_size//2  # 简单转换
            tgt = np.pad(tgt, (0, max_len - tgt_len), 'constant')
            
            self.data.append((src, tgt))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        src, tgt = self.data[idx]
        return torch.LongTensor(src), torch.LongTensor(tgt)


def train_translation_model():
    """训练翻译模型"""
    print("训练翻译模型...")
    print("=" * 50)
    
    # 参数设置
    vocab_size = 200  # 总词汇表大小
    d_model = 128
    n_layers = 4
    n_heads = 8
    d_ff = 512
    max_len = 30
    batch_size = 32
    num_epochs = 20
    learning_rate = 0.001
    
    # 创建数据集
    dataset = TranslationDataset(num_samples=1000, max_len=max_len, vocab_size=vocab_size)
    
    # 创建数据加载器
    from torch.utils.data import DataLoader, random_split
    
    # 划分训练集和验证集
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # 创建模型
    model = Transformer(
        src_vocab_size=vocab_size,
        tgt_vocab_size=vocab_size,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        d_ff=d_ff,
        max_len=max_len,
        dropout=0.1
    )
    
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 创建优化器和损失函数
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    
    # 训练循环
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        epoch_train_loss = 0
        
        for src, tgt in train_loader:
            optimizer.zero_grad()
            
            # 前向传播
            output = model(src, tgt[:, :-1])
            
            # 计算损失
            loss = criterion(
                output.reshape(-1, output.size(-1)),
                tgt[:, 1:].reshape(-1)
            )
            
            # 反向传播
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_train_loss += loss.item()
        
        # 验证阶段
        model.eval()
        epoch_val_loss = 0
        
        with torch.no_grad():
            for src, tgt in val_loader:
                output = model(src, tgt[:, :-1])
                loss = criterion(
                    output.reshape(-1, output.size(-1)),
                    tgt[:, 1:].reshape(-1)
                )
                epoch_val_loss += loss.item()
        
        # 计算平均损失
        avg_train_loss = epoch_train_loss / len(train_loader)
        avg_val_loss = epoch_val_loss / len(val_loader)
        
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        
        # 更新学习率
        scheduler.step()
        
        print(f"Epoch {epoch+1:2d}/{num_epochs}: "
              f"Train Loss: {avg_train_loss:.4f}, "
              f"Val Loss: {avg_val_loss:.4f}, "
              f"LR: {scheduler.get_last_lr()[0]:.6f}")
    
    return model, train_losses, val_losses


def translate_example(model, vocab_size=200):
    """翻译示例"""
    print("\n翻译示例...")
    print("=" * 50)
    
    # 创建测试句子
    test_src = torch.LongTensor([
        [10, 20, 30, 40, 50, 0, 0, 0],  # 英语句子
        [15, 25, 35, 0, 0, 0, 0, 0]     # 另一个句子
    ])
    
    print("源句子（英语）:")
    for i, src in enumerate(test_src):
        print(f"  句子{i+1}: {src.tolist()}")
    
    # 贪婪解码
    model.eval()
    with torch.no_grad():
        # 编码
        encoder_output = model.encode(test_src)
        
        # 初始化目标序列（开始标记为1）
        batch_size = test_src.size(0)
        tgt = torch.ones(batch_size, 1).fill_(1).long()
        
        for i in range(15):  # 最大生成长度
            # 解码
            output = model.decode(tgt, encoder_output)
            output = model.output_layer(output)
            
            # 获取下一个token
            next_token = output[:, -1:, :].argmax(-1)
            tgt = torch.cat([tgt, next_token], dim=1)
            
            # 检查是否结束（结束标记为2）
            if (next_token == 2).all():
                break
    
    print("\n翻译结果（法语）:")
    for i, translation in enumerate(tgt):
        print(f"  句子{i+1}: {translation.tolist()}")
    
    return tgt


def visualize_attention(model):
    """可视化注意力权重"""
    print("\n可视化注意力权重...")
    print("=" * 50)
    
    # 创建测试输入
    src = torch.LongTensor([[10, 20, 30, 40, 0, 0, 0, 0]])
    tgt = torch.LongTensor([[1, 65, 85, 105, 2, 0, 0, 0]])
    
    model.eval()
    with torch.no_grad():
        # 获取注意力权重
        src_mask, tgt_mask = model.generate_mask(src, tgt[:, :-1])
        encoder_output = model.encoder(src, src_mask)
        
        # 获取解码器第一层的注意力权重
        decoder_layer = model.decoder.layers[0]
        
        # 自注意力
        self_attn_output, self_attn_weights = decoder_layer.self_attn(
            model.decoder.token_embedding(tgt[:, :-1]),
            model.decoder.token_embedding(tgt[:, :-1]),
            model.decoder.token_embedding(tgt[:, :-1]),
            tgt_mask
        )
        
        # 交叉注意力
        cross_attn_output, cross_attn_weights = decoder_layer.cross_attn(
            self_attn_output,
            encoder_output,
            encoder_output,
            src_mask
        )
    
    print("自注意力权重形状:", self_attn_weights.shape)
    print("交叉注意力权重形状:", cross_attn_weights.shape)
    
    # 显示第一个头的注意力权重
    print("\n第一个头的自注意力权重（第一个样本）:")
    print(self_attn_weights[0, 0].detach().numpy().round(3))
    
    print("\n第一个头的交叉注意力权重（第一个样本）:")
    print(cross_attn_weights[0, 0].detach().numpy().round(3))


def main():
    """主函数"""
    print("高级Transformer示例：机器翻译")
    print("=" * 60)
    
    # 训练模型
    model, train_losses, val_losses = train_translation_model()
    
    # 翻译示例
    translations = translate_example(model)
    
    # 可视化注意力
    visualize_attention(model)
    
    # 保存模型
    torch.save(model.state_dict(), 'translation_model.pth')
    print("\n模型已保存到: translation_model.pth")
    
    print("\n✅ 高级示例完成！")


if __name__ == "__main__":
    main()