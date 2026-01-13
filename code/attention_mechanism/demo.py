#!/usr/bin/env python3
"""
注意力机制演示
展示如何使用不同的注意力机制
"""

import torch
import torch.nn as nn
from attention import (
    SelfAttention,
    CrossAttention,
    AdditiveAttention,
    PositionalEncoding,
    create_padding_mask
)


def demo_self_attention():
    """演示自注意力"""
    print("=" * 60)
    print("自注意力演示")
    print("=" * 60)
    
    # 参数设置
    batch_size = 1
    seq_len = 6
    d_model = 64
    n_heads = 4
    
    print(f"参数:")
    print(f"  batch_size: {batch_size}")
    print(f"  seq_len: {seq_len}")
    print(f"  d_model: {d_model}")
    print(f"  n_heads: {n_heads}")
    
    # 创建模型
    model = SelfAttention(d_model, n_heads)
    
    # 创建输入（模拟一个句子）
    # 假设每个词用d_model维向量表示
    input_sentence = torch.randn(batch_size, seq_len, d_model)
    
    print(f"\n输入句子形状: {input_sentence.shape}")
    print(f"（表示一个{batch_size}个句子，每个句子{seq_len}个词，每个词{d_model}维向量）")
    
    # 前向传播
    output, attention_weights = model(input_sentence)
    
    print(f"\n输出形状: {output.shape}")
    print(f"注意力权重形状: {attention_weights.shape}")
    
    # 分析注意力权重
    print(f"\n注意力权重分析:")
    print(f"  注意力头数量: {n_heads}")
    print(f"  每个头的注意力矩阵: {seq_len} × {seq_len}")
    
    # 显示第一个头的注意力权重
    first_head_weights = attention_weights[0, 0]  # [seq_len, seq_len]
    print(f"\n第一个头的注意力矩阵（前3×3）:")
    for i in range(min(3, seq_len)):
        row = [f"{first_head_weights[i, j].item():.3f}" for j in range(min(3, seq_len))]
        print(f"  行{i}: {row}")
    
    return model, input_sentence, output, attention_weights


def demo_cross_attention():
    """演示交叉注意力"""
    print("\n" + "=" * 60)
    print("交叉注意力演示")
    print("=" * 60)
    
    # 参数设置（模拟机器翻译：源语言到目标语言）
    batch_size = 1
    src_len = 8  # 源语言句子长度
    tgt_len = 6  # 目标语言句子长度
    d_model = 64
    n_heads = 4
    
    print(f"参数:")
    print(f"  batch_size: {batch_size}")
    print(f"  源语言长度: {src_len}")
    print(f"  目标语言长度: {tgt_len}")
    print(f"  d_model: {d_model}")
    print(f"  n_heads: {n_heads}")
    
    # 创建模型
    model = CrossAttention(d_model, n_heads)
    
    # 创建输入
    # query: 目标语言表示（解码器输出）
    # key_value: 源语言表示（编码器输出）
    query = torch.randn(batch_size, tgt_len, d_model)
    key_value = torch.randn(batch_size, src_len, d_model)
    
    print(f"\n查询（目标语言）形状: {query.shape}")
    print(f"键值（源语言）形状: {key_value.shape}")
    
    # 前向传播
    output, attention_weights = model(query, key_value)
    
    print(f"\n输出形状: {output.shape}")
    print(f"注意力权重形状: {attention_weights.shape}")
    
    # 分析注意力权重
    print(f"\n交叉注意力分析:")
    print(f"  每个目标词关注所有源词")
    print(f"  注意力矩阵: {tgt_len} × {src_len}")
    
    return model, query, key_value, output, attention_weights


def demo_additive_attention():
    """演示加性注意力"""
    print("\n" + "=" * 60)
    print("加性注意力演示（用于Seq2Seq模型）")
    print("=" * 60)
    
    # 参数设置
    batch_size = 1
    seq_len = 10  # 编码器输出序列长度
    hidden_size = 128
    
    print(f"参数:")
    print(f"  batch_size: {batch_size}")
    print(f"  序列长度: {seq_len}")
    print(f"  隐藏层大小: {hidden_size}")
    
    # 创建模型
    model = AdditiveAttention(hidden_size)
    
    # 创建输入
    # query: 解码器当前隐藏状态
    # keys: 编码器所有隐藏状态
    query = torch.randn(batch_size, hidden_size)
    keys = torch.randn(batch_size, seq_len, hidden_size)
    
    print(f"\n查询（解码器隐藏状态）形状: {query.shape}")
    print(f"键（编码器隐藏状态）形状: {keys.shape}")
    
    # 前向传播
    context, attention_weights = model(query, keys)
    
    print(f"\n上下文向量形状: {context.shape}")
    print(f"注意力权重形状: {attention_weights.shape}")
    
    # 显示注意力权重
    print(f"\n注意力权重（每个时间步的重要性）:")
    weights = attention_weights[0]  # [seq_len]
    for i in range(min(5, seq_len)):
        print(f"  时间步{i}: {weights[i].item():.4f}")
    
    return model, query, keys, context, attention_weights


def demo_positional_encoding():
    """演示位置编码"""
    print("\n" + "=" * 60)
    print("位置编码演示")
    print("=" * 60)
    
    # 参数设置
    batch_size = 1
    seq_len = 8
    d_model = 32
    
    print(f"参数:")
    print(f"  batch_size: {batch_size}")
    print(f"  序列长度: {seq_len}")
    print(f"  模型维度: {d_model}")
    
    # 创建模型
    model = PositionalEncoding(d_model)
    
    # 创建输入（没有位置信息的词嵌入）
    embeddings = torch.randn(batch_size, seq_len, d_model)
    
    print(f"\n词嵌入形状（无位置信息）: {embeddings.shape}")
    
    # 添加位置编码
    embeddings_with_pe = model(embeddings)
    
    print(f"带位置编码的词嵌入形状: {embeddings_with_pe.shape}")
    
    # 显示位置编码的效果
    print(f"\n位置编码效果:")
    print(f"  原始嵌入范数: {embeddings.norm():.4f}")
    print(f"  带位置编码的嵌入范数: {embeddings_with_pe.norm():.4f}")
    print(f"  位置编码范数: {(embeddings_with_pe - embeddings).norm():.4f}")
    
    return model, embeddings, embeddings_with_pe


def demo_masked_attention():
    """演示带掩码的注意力"""
    print("\n" + "=" * 60)
    print("带掩码的注意力演示")
    print("=" * 60)
    
    # 参数设置
    batch_size = 1
    seq_len = 6
    d_model = 64
    n_heads = 4
    
    print(f"参数:")
    print(f"  batch_size: {batch_size}")
    print(f"  序列长度: {seq_len}")
    print(f"  d_model: {d_model}")
    print(f"  n_heads: {n_heads}")
    
    # 创建模型
    model = SelfAttention(d_model, n_heads)
    
    # 创建输入
    input_tensor = torch.randn(batch_size, seq_len, d_model)
    
    # 创建掩码（模拟填充：前4个是真实词，后2个是填充）
    seq = torch.tensor([[1, 1, 1, 1, 0, 0]])  # 1表示真实词，0表示填充
    mask = create_padding_mask(seq, pad_token_id=0)
    
    print(f"\n输入序列: {seq.tolist()}")
    print(f"（1=真实词，0=填充）")
    print(f"掩码形状: {mask.shape}")
    
    # 显示掩码
    mask_display = mask.squeeze().int()
    print(f"掩码矩阵:")
    for i in range(seq_len):
        row = [str(mask_display[i, j].item()) for j in range(seq_len)]
        print(f"  行{i}: {row}")
    
    # 前向传播（带掩码）
    output, attention_weights = model(input_tensor, mask=mask)
    
    print(f"\n输出形状: {output.shape}")
    print(f"注意力权重形状: {attention_weights.shape}")
    
    # 检查掩码效果
    print(f"\n掩码效果验证:")
    print(f"  填充位置的注意力权重应该接近0")
    
    # 检查第一个头的注意力权重
    first_head = attention_weights[0, 0]  # [seq_len, seq_len]
    
    # 检查填充位置（索引4,5）的注意力权重
    for i in [4, 5]:  # 填充位置
        row_sum = first_head[i].sum().item()
        print(f"  行{i}（填充位置）的注意力权重总和: {row_sum:.6f}")
    
    for i in [0, 1]:  # 真实词位置
        row_sum = first_head[i].sum().item()
        print(f"  行{i}（真实词位置）的注意力权重总和: {row_sum:.4f}")
    
    return model, input_tensor, mask, output, attention_weights


def demo_attention_visualization():
    """演示注意力可视化"""
    print("\n" + "=" * 60)
    print("注意力可视化演示")
    print("=" * 60)
    
    # 创建一个简单的例子
    seq_len = 5
    d_model = 16
    n_heads = 2
    
    # 创建模型
    model = SelfAttention(d_model, n_heads)
    
    # 创建有意义的输入（模拟一个句子）
    # 让某些词之间有更强的关联
    input_tensor = torch.randn(1, seq_len, d_model)
    
    # 手动增强某些关联
    # 让词0和词2相似
    input_tensor[0, 2] = input_tensor[0, 0] + 0.1 * torch.randn(d_model)
    # 让词1和词3相似
    input_tensor[0, 3] = input_tensor[0, 1] + 0.1 * torch.randn(d_model)
    
    # 前向传播
    output, attention_weights = model(input_tensor)
    
    print(f"序列长度: {seq_len}")
    print(f"注意力头数量: {n_heads}")
    
    # 显示每个头的注意力模式
    for head_idx in range(n_heads):
        print(f"\n注意力头 {head_idx}:")
        weights = attention_weights[0, head_idx]  # [seq_len, seq_len]
        
        # 创建简单的文本表示
        print("  注意力矩阵:")
        for i in range(seq_len):
            row = [f"{weights[i, j].item():.2f}" for j in range(seq_len)]
            print(f"    词{i} → {row}")
        
        # 找出每个词最关注的词
        print(f"  每个词最关注的词:")
        for i in range(seq_len):
            max_attention_idx = weights[i].argmax().item()
            max_attention_value = weights[i, max_attention_idx].item()
            print(f"    词{i} → 词{max_attention_idx} ({max_attention_value:.3f})")
    
    return attention_weights


def main():
    """主演示函数"""
    print("注意力机制完整演示")
    print("=" * 60)
    
    try:
        # 运行所有演示
        demo_self_attention()
        demo_cross_attention()
        demo_additive_attention()
        demo_positional_encoding()
        demo_masked_attention()
        demo_attention_visualization()
        
        print("\n" + "=" * 60)
        print("演示总结")
        print("=" * 60)
        print("""
已演示的注意力机制类型：
1. 自注意力 (Self-Attention)
   - 输入序列中的每个元素关注序列中的所有元素
   - 用于捕捉序列内部的依赖关系
   
2. 交叉注意力 (Cross-Attention)
   - 一个序列（查询）关注另一个序列（键值）
   - 用于机器翻译、问答系统等
   
3. 加性注意力 (Additive Attention)
   - 使用加性方式计算注意力分数
   - 常用于Seq2Seq模型的注意力机制
   
4. 位置编码 (Positional Encoding)
   - 为Transformer添加序列位置信息
   - 使用正弦和余弦函数
   
5. 带掩码的注意力 (Masked Attention)
   - 使用掩码控制注意力范围
   - 用于处理填充、实现因果注意力等
   
6. 注意力可视化
   - 分析和理解注意力模式
   - 帮助调试和解释模型行为
        """)
        
        print("\n使用建议：")
        print("1. 自注意力：用于文本分类、语言建模等任务")
        print("2. 交叉注意力：用于机器翻译、文本摘要等任务")
        print("3. 加性注意力：用于传统的Seq2Seq模型")
        print("4. 多头注意力：增加模型容量，捕捉不同方面的信息")
        
        return True
        
    except Exception as e:
        print(f"演示失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("开始注意力机制演示...")
    success = main()
    if success:
        print("\n演示完成！")
    else:
        print("\n演示失败！")