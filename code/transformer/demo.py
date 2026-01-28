"""
Transformer演示：展示各个组件的功能
"""

import torch
import torch.nn as nn
import numpy as np
from transformer import (
    PositionalEncoding, 
    MultiHeadAttention, 
    PositionwiseFeedForward,
    EncoderLayer,
    DecoderLayer,
    Transformer,
    create_sample_data
)


def demo_positional_encoding():
    """演示位置编码"""
    print("=" * 50)
    print("1. 位置编码演示")
    print("=" * 50)
    
    d_model = 16
    max_len = 10
    pe = PositionalEncoding(d_model, max_len)
    
    # 创建输入
    batch_size = 2
    seq_len = 5
    x = torch.randn(seq_len, batch_size, d_model)
    
    # 应用位置编码
    output = pe(x)
    
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")
    print(f"位置编码形状: {pe.pe.shape}")
    
    # 可视化位置编码
    print("\n位置编码矩阵（前5个位置，前8个维度）:")
    print(pe.pe[:5, 0, :8].detach().numpy())
    
    return pe


def demo_multihead_attention():
    """演示多头注意力"""
    print("\n" + "=" * 50)
    print("2. 多头注意力演示")
    print("=" * 50)
    
    d_model = 16
    n_heads = 4
    batch_size = 2
    seq_len = 6
    
    # 创建多头注意力层
    mha = MultiHeadAttention(d_model, n_heads)
    
    # 创建输入
    query = torch.randn(batch_size, seq_len, d_model)
    key = torch.randn(batch_size, seq_len, d_model)
    value = torch.randn(batch_size, seq_len, d_model)
    
    # 创建mask（模拟padding）
    mask = torch.ones(batch_size, seq_len, seq_len)
    mask[:, :, 4:] = 0  # 最后两个位置是padding
    
    # 前向传播
    output, attn_weights = mha(query, key, value, mask)
    
    print(f"查询形状: {query.shape}")
    print(f"键形状: {key.shape}")
    print(f"值形状: {value.shape}")
    print(f"输出形状: {output.shape}")
    print(f"注意力权重形状: {attn_weights.shape}")
    
    # 展示注意力权重
    print(f"\n第一个样本的第一个头的注意力权重:")
    print(attn_weights[0, 0].detach().numpy())
    
    return mha


def demo_feed_forward():
    """演示前馈网络"""
    print("\n" + "=" * 50)
    print("3. 前馈网络演示")
    print("=" * 50)
    
    d_model = 16
    d_ff = 32
    batch_size = 2
    seq_len = 5
    
    # 创建前馈网络
    ff = PositionwiseFeedForward(d_model, d_ff)
    
    # 创建输入
    x = torch.randn(batch_size, seq_len, d_model)
    
    # 前向传播
    output = ff(x)
    
    print(f"输入形状: {x.shape}")
    print(f)输出形状: {output.shape}")
    print(f"参数数量:")
    print(f"  linear1: {sum(p.numel() for p in ff.linear1.parameters())}")
    print(f"  linear2: {sum(p.numel() for p in ff.linear2.parameters())}")
    
    return ff


def demo_encoder_layer():
    """演示编码器层"""
    print("\n" + "=" * 50)
    print("4. 编码器层演示")
    print("=" * 50)
    
    d_model = 16
    n_heads = 4
    d_ff = 32
    batch_size = 2
    seq_len = 6
    
    # 创建编码器层
    encoder_layer = EncoderLayer(d_model, n_heads, d_ff)
    
    # 创建输入
    x = torch.randn(batch_size, seq_len, d_model)
    
    # 创建mask
    mask = torch.ones(batch_size, 1, 1, seq_len)
    mask[:, :, :, 4:] = 0  # 最后两个位置是padding
    
    # 前向传播
    output = encoder_layer(x, mask)
    
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")
    
    # 检查残差连接
    print(f"\n输入输出差异范数: {torch.norm(output - x):.4f}")
    
    return encoder_layer


def demo_decoder_layer():
    """演示解码器层"""
    print("\n" + "=" * 50)
    print("5. 解码器层演示")
    print("=" * 50)
    
    d_model = 16
    n_heads = 4
    d_ff = 32
    batch_size = 2
    seq_len = 6
    
    # 创建解码器层
    decoder_layer = DecoderLayer(d_model, n_heads, d_ff)
    
    # 创建输入
    x = torch.randn(batch_size, seq_len, d_model)
    encoder_output = torch.randn(batch_size, seq_len, d_model)
    
    # 创建mask
    src_mask = torch.ones(batch_size, 1, 1, seq_len)
    src_mask[:, :, :, 4:] = 0
    
    # 目标序列mask（look-ahead mask）
    tgt_mask = torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0).unsqueeze(0)
    
    # 前向传播
    output = decoder_layer(x, encoder_output, src_mask, tgt_mask)
    
    print(f"解码器输入形状: {x.shape}")
    print(f"编码器输出形状: {encoder_output.shape}")
    print(f"解码器输出形状: {output.shape}")
    
    return decoder_layer


def demo_full_transformer():
    """演示完整Transformer"""
    print("\n" + "=" * 50)
    print("6. 完整Transformer演示")
    print("=" * 50)
    
    # 创建小型的Transformer
    model = Transformer(
        src_vocab_size=50,
        tgt_vocab_size=50,
        d_model=32,
        n_heads=4,
        num_encoder_layers=2,
        num_decoder_layers=2,
        d_ff=64,
        max_seq_length=15
    )
    
    # 创建示例数据
    src, tgt = create_sample_data(vocab_size=50, batch_size=3, seq_len=8)
    
    print(f"源序列形状: {src.shape}")
    print(f"目标序列形状: {tgt.shape}")
    
    # 前向传播
    output = model(src, tgt[:, :-1])
    
    print(f"模型输出形状: {output.shape}")
    
    # 计算损失
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    loss = criterion(output.reshape(-1, 50), tgt[:, 1:].reshape(-1))
    
    print(f"交叉熵损失: {loss.item():.4f}")
    
    # 生成示例
    print("\n生成示例:")
    generated = model.generate(src, max_len=12)
    print(f"生成序列形状: {generated.shape}")
    
    # 展示一个生成结果
    print(f"第一个样本的生成结果:")
    print(f"源序列: {src[0].numpy()}")
    print(f"生成序列: {generated[0].numpy()}")
    
    return model


def main():
    """主演示函数"""
    print("Transformer组件演示")
    print("=" * 50)
    
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 演示各个组件
    pe = demo_positional_encoding()
    mha = demo_multihead_attention()
    ff = demo_feed_forward()
    encoder_layer = demo_encoder_layer()
    decoder_layer = demo_decoder_layer()
    transformer = demo_full_transformer()
    
    print("\n" + "=" * 50)
    print("演示完成！")
    print("=" * 50)
    
    # 总结
    print("\n组件总结:")
    print(f"1. 位置编码: 将位置信息注入到输入中")
    print(f"2. 多头注意力: 并行计算多个注意力头")
    print(f"3. 前馈网络: 每个位置独立的前馈网络")
    print(f"4. 编码器层: 自注意力 + 前馈网络 + 残差连接")
    print(f"5. 解码器层: 自注意力 + 交叉注意力 + 前馈网络 + 残差连接")
    print(f"6. 完整Transformer: 编码器-解码器架构")
    
    return {
        'positional_encoding': pe,
        'multihead_attention': mha,
        'feed_forward': ff,
        'encoder_layer': encoder_layer,
        'decoder_layer': decoder_layer,
        'transformer': transformer
    }


if __name__ == "__main__":
    components = main()