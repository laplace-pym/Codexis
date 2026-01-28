import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from transformer_model import Transformer

def create_sample_data(vocab_size=100, batch_size=4, seq_len=10):
    """创建示例数据"""
    # 创建源序列和目标序列
    src = torch.randint(3, vocab_size-1, (batch_size, seq_len))
    tgt = torch.randint(3, vocab_size-1, (batch_size, seq_len))
    
    # 添加开始和结束标记
    src[:, 0] = 1  # 开始标记
    tgt[:, 0] = 1  # 开始标记
    src[:, -1] = 2  # 结束标记
    tgt[:, -1] = 2  # 结束标记
    
    return src, tgt

def test_transformer():
    """测试Transformer模型"""
    print("测试Transformer模型...")
    
    # 超参数
    src_vocab_size = 100
    tgt_vocab_size = 100
    d_model = 128
    num_heads = 8
    num_encoder_layers = 3
    num_decoder_layers = 3
    d_ff = 512
    batch_size = 4
    seq_len = 10
    
    # 创建模型
    model = Transformer(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
        d_ff=d_ff,
        max_seq_len=seq_len,
        dropout=0.1
    )
    
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 创建示例数据
    src, tgt = create_sample_data(src_vocab_size, batch_size, seq_len)
    
    print(f"源序列形状: {src.shape}")
    print(f"目标序列形状: {tgt.shape}")
    
    # 前向传播
    output = model(src, tgt[:, :-1])  # 训练时使用shifted right
    
    print(f"输出形状: {output.shape}")
    print(f"输出范围: [{output.min():.4f}, {output.max():.4f}]")
    
    # 测试生成
    generated = model.generate(src, max_len=15)
    print(f"生成序列形状: {generated.shape}")
    print(f"生成序列示例: {generated[0]}")
    
    return model, src, tgt, output

def train_simple_example():
    """训练一个简单的示例"""
    print("\n训练简单示例...")
    
    # 超参数
    src_vocab_size = 50
    tgt_vocab_size = 50
    d_model = 64
    num_heads = 4
    num_encoder_layers = 2
    num_decoder_layers = 2
    d_ff = 256
    batch_size = 8
    seq_len = 8
    num_epochs = 10
    learning_rate = 0.001
    
    # 创建模型
    model = Transformer(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
        d_ff=d_ff,
        max_seq_len=seq_len,
        dropout=0.1
    )
    
    # 优化器和损失函数
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # 忽略填充标记
    
    # 训练循环
    for epoch in range(num_epochs):
        model.train()
        
        # 创建训练数据
        src, tgt = create_sample_data(src_vocab_size, batch_size, seq_len)
        
        # 前向传播
        output = model(src, tgt[:, :-1])
        
        # 计算损失
        loss = criterion(
            output.reshape(-1, tgt_vocab_size),
            tgt[:, 1:].reshape(-1)  # 使用shifted right作为目标
        )
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 2 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")
    
    print("训练完成!")
    
    # 测试生成
    model.eval()
    test_src, _ = create_sample_data(src_vocab_size, 2, seq_len)
    generated = model.generate(test_src, max_len=12)
    
    print(f"\n测试生成:")
    print(f"源序列: {test_src[0]}")
    print(f"生成序列: {generated[0]}")

def test_attention_mechanism():
    """测试注意力机制"""
    print("\n测试注意力机制...")
    
    from transformer_model import MultiHeadAttention
    
    # 创建多头注意力层
    d_model = 64
    num_heads = 4
    batch_size = 2
    seq_len = 5
    
    attention = MultiHeadAttention(d_model, num_heads)
    
    # 创建输入
    Q = torch.randn(batch_size, seq_len, d_model)
    K = torch.randn(batch_size, seq_len, d_model)
    V = torch.randn(batch_size, seq_len, d_model)
    
    # 前向传播
    output, attn_probs = attention(Q, K, V)
    
    print(f"查询(Q)形状: {Q.shape}")
    print(f"键(K)形状: {K.shape}")
    print(f"值(V)形状: {V.shape}")
    print(f"注意力输出形状: {output.shape}")
    print(f"注意力权重形状: {attn_probs.shape}")
    
    # 检查注意力权重是否和为1
    attn_sum = attn_probs.sum(dim=-1)
    print(f"注意力权重每行和: {attn_sum[0, 0]}")

if __name__ == "__main__":
    print("=" * 60)
    print("Transformer模型演示")
    print("=" * 60)
    
    # 测试基本功能
    model, src, tgt, output = test_transformer()
    
    # 测试注意力机制
    test_attention_mechanism()
    
    # 训练简单示例
    train_simple_example()
    
    print("\n" + "=" * 60)
    print("演示完成!")
    print("=" * 60)