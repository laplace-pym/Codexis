#!/usr/bin/env python3
"""
测试注意力机制代码
"""

import torch
import torch.nn as nn
from attention import (
    ScaledDotProductAttention,
    MultiHeadAttention,
    SelfAttention,
    CrossAttention,
    AdditiveAttention,
    PositionalEncoding,
    create_padding_mask,
    create_look_ahead_mask
)


def test_scaled_dot_product_attention():
    """测试缩放点积注意力"""
    print("测试缩放点积注意力...")
    
    batch_size = 2
    n_heads = 4
    seq_len = 5
    d_k = 64
    d_v = 64
    
    # 创建随机输入
    query = torch.randn(batch_size, n_heads, seq_len, d_k)
    key = torch.randn(batch_size, n_heads, seq_len, d_k)
    value = torch.randn(batch_size, n_heads, seq_len, d_v)
    
    # 创建注意力层
    attention = ScaledDotProductAttention(dropout=0.1)
    
    # 前向传播
    output, attention_weights = attention(query, key, value)
    
    print(f"查询形状: {query.shape}")
    print(f"键形状: {key.shape}")
    print(f"值形状: {value.shape}")
    print(f"输出形状: {output.shape}")
    print(f"注意力权重形状: {attention_weights.shape}")
    
    # 验证形状
    assert output.shape == (batch_size, n_heads, seq_len, d_v)
    assert attention_weights.shape == (batch_size, n_heads, seq_len, seq_len)
    
    print("✓ 缩放点积注意力测试通过\n")
    return True


def test_multihead_attention():
    """测试多头注意力"""
    print("测试多头注意力...")
    
    batch_size = 2
    seq_len = 10
    d_model = 512
    n_heads = 8
    
    # 创建随机输入
    query = torch.randn(batch_size, seq_len, d_model)
    key = torch.randn(batch_size, seq_len, d_model)
    value = torch.randn(batch_size, seq_len, d_model)
    
    # 创建多头注意力层
    multihead_attention = MultiHeadAttention(d_model, n_heads, dropout=0.1)
    
    # 前向传播
    output, attention_weights = multihead_attention(query, key, value)
    
    print(f"查询形状: {query.shape}")
    print(f"键形状: {key.shape}")
    print(f"值形状: {value.shape}")
    print(f"输出形状: {output.shape}")
    print(f"注意力权重形状: {attention_weights.shape}")
    
    # 验证形状
    assert output.shape == (batch_size, seq_len, d_model)
    assert attention_weights.shape == (batch_size, n_heads, seq_len, seq_len)
    
    print("✓ 多头注意力测试通过\n")
    return True


def test_self_attention():
    """测试自注意力"""
    print("测试自注意力...")
    
    batch_size = 2
    seq_len = 10
    d_model = 512
    n_heads = 8
    
    # 创建随机输入
    x = torch.randn(batch_size, seq_len, d_model)
    
    # 创建自注意力层
    self_attention = SelfAttention(d_model, n_heads, dropout=0.1)
    
    # 前向传播
    output, attention_weights = self_attention(x)
    
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")
    print(f"注意力权重形状: {attention_weights.shape}")
    
    # 验证形状
    assert output.shape == (batch_size, seq_len, d_model)
    assert attention_weights.shape == (batch_size, n_heads, seq_len, seq_len)
    
    print("✓ 自注意力测试通过\n")
    return True


def test_cross_attention():
    """测试交叉注意力"""
    print("测试交叉注意力...")
    
    batch_size = 2
    seq_len_q = 5
    seq_len_kv = 10
    d_model = 512
    n_heads = 8
    
    # 创建随机输入
    query = torch.randn(batch_size, seq_len_q, d_model)
    key_value = torch.randn(batch_size, seq_len_kv, d_model)
    
    # 创建交叉注意力层
    cross_attention = CrossAttention(d_model, n_heads, dropout=0.1)
    
    # 前向传播
    output, attention_weights = cross_attention(query, key_value)
    
    print(f"查询形状: {query.shape}")
    print(f"键值形状: {key_value.shape}")
    print(f"输出形状: {output.shape}")
    print(f"注意力权重形状: {attention_weights.shape}")
    
    # 验证形状
    assert output.shape == (batch_size, seq_len_q, d_model)
    assert attention_weights.shape == (batch_size, n_heads, seq_len_q, seq_len_kv)
    
    print("✓ 交叉注意力测试通过\n")
    return True


def test_additive_attention():
    """测试加性注意力"""
    print("测试加性注意力...")
    
    batch_size = 2
    seq_len = 10
    hidden_size = 256
    
    # 创建随机输入
    query = torch.randn(batch_size, hidden_size)
    keys = torch.randn(batch_size, seq_len, hidden_size)
    
    # 创建加性注意力层
    additive_attention = AdditiveAttention(hidden_size)
    
    # 前向传播
    context, attention_weights = additive_attention(query, keys)
    
    print(f"查询形状: {query.shape}")
    print(f"键形状: {keys.shape}")
    print(f"上下文向量形状: {context.shape}")
    print(f"注意力权重形状: {attention_weights.shape}")
    
    # 验证形状
    assert context.shape == (batch_size, hidden_size)
    assert attention_weights.shape == (batch_size, seq_len)
    
    print("✓ 加性注意力测试通过\n")
    return True


def test_positional_encoding():
    """测试位置编码"""
    print("测试位置编码...")
    
    batch_size = 2
    seq_len = 20
    d_model = 512
    
    # 创建随机输入
    x = torch.randn(batch_size, seq_len, d_model)
    
    # 创建位置编码层
    positional_encoding = PositionalEncoding(d_model)
    
    # 前向传播
    output = positional_encoding(x)
    
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")
    
    # 验证形状
    assert output.shape == (batch_size, seq_len, d_model)
    
    print("✓ 位置编码测试通过\n")
    return True


def test_masks():
    """测试掩码"""
    print("测试掩码...")
    
    # 测试填充掩码
    seq = torch.tensor([[1, 2, 3, 0, 0], [1, 2, 0, 0, 0]])
    padding_mask = create_padding_mask(seq, pad_token_id=0)
    
    print(f"序列: {seq}")
    print(f"填充掩码形状: {padding_mask.shape}")
    print(f"填充掩码: {padding_mask.squeeze().int()}")
    
    # 测试前瞻掩码
    size = 5
    look_ahead_mask = create_look_ahead_mask(size)
    
    print(f"\n前瞻掩码形状: {look_ahead_mask.shape}")
    print(f"前瞻掩码:\n{look_ahead_mask.int()}")
    
    print("✓ 掩码测试通过\n")
    return True


def test_attention_with_masks():
    """测试带掩码的注意力"""
    print("测试带掩码的注意力...")
    
    batch_size = 2
    seq_len = 5
    d_model = 512
    n_heads = 8
    
    # 创建随机输入
    x = torch.randn(batch_size, seq_len, d_model)
    
    # 创建填充掩码
    seq = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 0, 0, 0]])
    padding_mask = create_padding_mask(seq, pad_token_id=0)
    
    # 创建自注意力层
    self_attention = SelfAttention(d_model, n_heads, dropout=0.1)
    
    # 前向传播（带掩码）
    output, attention_weights = self_attention(x, mask=padding_mask)
    
    print(f"输入形状: {x.shape}")
    print(f"填充掩码形状: {padding_mask.shape}")
    print(f"输出形状: {output.shape}")
    print(f"注意力权重形状: {attention_weights.shape}")
    
    # 验证形状
    assert output.shape == (batch_size, seq_len, d_model)
    assert attention_weights.shape == (batch_size, n_heads, seq_len, seq_len)
    
    print("✓ 带掩码的注意力测试通过\n")
    return True


def test_transformer_style_attention():
    """测试Transformer风格的注意力"""
    print("测试Transformer风格的注意力...")
    
    batch_size = 2
    seq_len = 10
    d_model = 512
    n_heads = 8
    
    # 创建随机输入
    x = torch.randn(batch_size, seq_len, d_model)
    
    # 创建位置编码
    positional_encoding = PositionalEncoding(d_model)
    x_with_pe = positional_encoding(x)
    
    # 创建自注意力层
    self_attention = SelfAttention(d_model, n_heads, dropout=0.1)
    
    # 创建前瞻掩码
    look_ahead_mask = create_look_ahead_mask(seq_len)
    look_ahead_mask = look_ahead_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, seq_len]
    
    # 前向传播
    output, attention_weights = self_attention(x_with_pe, mask=look_ahead_mask)
    
    print(f"输入形状: {x.shape}")
    print(f"带位置编码的输入形状: {x_with_pe.shape}")
    print(f"输出形状: {output.shape}")
    print(f"注意力权重形状: {attention_weights.shape}")
    
    # 验证形状
    assert output.shape == (batch_size, seq_len, d_model)
    
    print("✓ Transformer风格注意力测试通过\n")
    return True


def main():
    """主测试函数"""
    print("=" * 60)
    print("开始测试注意力机制")
    print("=" * 60)
    
    try:
        # 运行所有测试
        test_scaled_dot_product_attention()
        test_multihead_attention()
        test_self_attention()
        test_cross_attention()
        test_additive_attention()
        test_positional_encoding()
        test_masks()
        test_attention_with_masks()
        test_transformer_style_attention()
        
        print("=" * 60)
        print("所有测试通过！")
        print("=" * 60)
        
        # 演示如何使用注意力机制
        print("\n" + "=" * 60)
        print("注意力机制使用示例")
        print("=" * 60)
        
        # 示例1：创建简单的自注意力模型
        print("\n示例1：创建简单的自注意力模型")
        d_model = 256
        n_heads = 4
        seq_len = 8
        
        model = SelfAttention(d_model, n_heads)
        input_tensor = torch.randn(1, seq_len, d_model)
        output, attention_weights = model(input_tensor)
        
        print(f"输入: {input_tensor.shape}")
        print(f"输出: {output.shape}")
        print(f"注意力权重: {attention_weights.shape}")
        
        # 示例2：可视化注意力权重
        print("\n示例2：注意力权重统计")
        print(f"注意力权重最小值: {attention_weights.min().item():.4f}")
        print(f"注意力权重最大值: {attention_weights.max().item():.4f}")
        print(f"注意力权重平均值: {attention_weights.mean().item():.4f}")
        print(f"注意力权重总和（每行）: {attention_weights.sum(dim=-1).mean().item():.4f}")
        
        # 示例3：使用掩码
        print("\n示例3：使用填充掩码")
        seq = torch.tensor([[1, 2, 3, 4, 0, 0, 0, 0]])
        mask = create_padding_mask(seq, pad_token_id=0)
        output_masked, attention_weights_masked = model(input_tensor, mask=mask)
        print(f"掩码形状: {mask.shape}")
        print(f"带掩码的输出形状: {output_masked.shape}")
        
        return True
        
    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    if success:
        print("\n注意力机制代码测试完成！")
    else:
        print("\n注意力机制代码测试失败！")