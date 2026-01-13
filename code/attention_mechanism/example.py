"""
注意力机制使用示例
"""

import torch
import torch.nn as nn
from attention import (
    ScaledDotProductAttention,
    MultiHeadAttention,
    SelfAttention,
    CrossAttention,
    AdditiveAttention,
    CausalAttention,
    PositionalEncoding,
    TransformerBlock
)


def basic_attention_example():
    """基础注意力机制示例"""
    print("=" * 50)
    print("基础注意力机制示例")
    print("=" * 50)
    
    # 创建测试数据
    batch_size = 2
    seq_len = 4
    d_model = 8
    
    # 输入序列
    x = torch.randn(batch_size, seq_len, d_model)
    print(f"输入序列形状: {x.shape}")
    print(f"输入序列:\n{x[0]}")
    
    # 1. 缩放点积注意力
    print("\n1. 缩放点积注意力:")
    attention = ScaledDotProductAttention()
    output, weights = attention(x, x, x)
    print(f"输出形状: {output.shape}")
    print(f"注意力权重形状: {weights.shape}")
    print(f"第一个样本的注意力权重:\n{weights[0]}")
    
    # 2. 多头注意力
    print("\n2. 多头注意力 (4个头):")
    multi_head = MultiHeadAttention(d_model, num_heads=4)
    output = multi_head(x, x, x)
    print(f"输出形状: {output.shape}")
    
    return x, output


def self_attention_example():
    """自注意力示例"""
    print("\n" + "=" * 50)
    print("自注意力示例")
    print("=" * 50)
    
    # 创建测试数据
    batch_size = 1
    seq_len = 6
    d_model = 16
    
    # 句子编码
    sentence = torch.randn(batch_size, seq_len, d_model)
    
    # 创建自注意力模型
    self_attn = SelfAttention(d_model, num_heads=4)
    
    # 应用自注意力
    output = self_attn(sentence)
    
    print(f"输入句子编码形状: {sentence.shape}")
    print(f"自注意力输出形状: {output.shape}")
    
    # 可视化注意力权重（简化版）
    print("\n注意力机制帮助模型理解句子中单词之间的关系")
    print("例如：在句子 'The cat sat on the mat' 中")
    print("'sat' 应该与 'cat' 有较强的注意力连接")
    
    return sentence, output


def cross_attention_example():
    """交叉注意力示例（用于机器翻译等任务）"""
    print("\n" + "=" * 50)
    print("交叉注意力示例（机器翻译场景）")
    print("=" * 50)
    
    # 模拟机器翻译场景
    batch_size = 1
    source_len = 5  # 源语言序列长度
    target_len = 4  # 目标语言序列长度
    d_model = 32
    
    # 源语言编码（编码器输出）
    source_encoding = torch.randn(batch_size, source_len, d_model)
    
    # 目标语言编码（解码器输入）
    target_encoding = torch.randn(batch_size, target_len, d_model)
    
    # 创建交叉注意力模型
    cross_attn = CrossAttention(d_model, num_heads=4)
    
    # 应用交叉注意力
    # 查询来自目标语言，键值来自源语言
    output = cross_attn(target_encoding, source_encoding)
    
    print(f"源语言编码形状: {source_encoding.shape}")
    print(f"目标语言编码形状: {target_encoding.shape}")
    print(f"交叉注意力输出形状: {output.shape}")
    
    print("\n交叉注意力机制允许解码器在生成每个目标词时")
    print("关注源语言中最相关的部分")
    
    return source_encoding, target_encoding, output


def causal_attention_example():
    """因果注意力示例（用于文本生成）"""
    print("\n" + "=" * 50)
    print("因果注意力示例（文本生成场景）")
    print("=" * 50)
    
    # 模拟文本生成
    batch_size = 1
    seq_len = 8
    d_model = 16
    
    # 已生成的文本编码
    generated_text = torch.randn(batch_size, seq_len, d_model)
    
    # 创建因果注意力模型
    causal_attn = CausalAttention()
    
    # 应用因果注意力
    output, weights = causal_attn(generated_text, generated_text, generated_text)
    
    print(f"已生成文本编码形状: {generated_text.shape}")
    print(f"因果注意力输出形状: {output.shape}")
    print(f"注意力权重形状: {weights.shape}")
    
    # 检查因果掩码
    print("\n因果注意力权重矩阵（应该是下三角矩阵）:")
    print(weights[0])
    
    print("\n因果注意力确保每个位置只能关注之前的位置")
    print("这对于自回归文本生成是必要的")
    
    return generated_text, output


def transformer_example():
    """Transformer块示例"""
    print("\n" + "=" * 50)
    print("Transformer块示例")
    print("=" * 50)
    
    # 创建测试数据
    batch_size = 2
    seq_len = 10
    d_model = 64
    
    # 输入序列
    x = torch.randn(batch_size, seq_len, d_model)
    
    # 添加位置编码
    pos_encoder = PositionalEncoding(d_model)
    x_with_pos = pos_encoder(x)
    
    # 创建Transformer块
    transformer = TransformerBlock(d_model, num_heads=8, d_ff=128)
    
    # 应用Transformer块
    output = transformer(x_with_pos)
    
    print(f"输入形状: {x.shape}")
    print(f"添加位置编码后形状: {x_with_pos.shape}")
    print(f"Transformer块输出形状: {output.shape}")
    
    print("\nTransformer块包含:")
    print("1. 多头自注意力")
    print("2. 前馈神经网络")
    print("3. 残差连接和层归一化")
    
    return x, output


def attention_visualization():
    """注意力可视化示例"""
    print("\n" + "=" * 50)
    print("注意力可视化示例")
    print("=" * 50)
    
    # 创建一个简单的句子对
    source_sentence = ["I", "love", "machine", "learning"]
    target_sentence = ["我", "喜欢", "机器", "学习"]
    
    # 模拟注意力权重
    attention_weights = torch.tensor([
        [0.8, 0.1, 0.05, 0.05],  # "我" 关注 "I"
        [0.1, 0.7, 0.1, 0.1],    # "喜欢" 关注 "love"
        [0.05, 0.1, 0.7, 0.15],  # "机器" 关注 "machine"
        [0.05, 0.1, 0.15, 0.7],  # "学习" 关注 "learning"
    ])
    
    print("源句子:", " ".join(source_sentence))
    print("目标句子:", " ".join(target_sentence))
    print("\n注意力权重矩阵:")
    print(attention_weights)
    
    print("\n注意力可视化解释:")
    print("每一行表示目标词对源词的关注程度")
    print("对角线附近的值较大，表示对齐关系")
    
    return attention_weights


def main():
    """主函数：运行所有示例"""
    print("注意力机制代码示例")
    print("=" * 60)
    
    # 运行各个示例
    basic_attention_example()
    self_attention_example()
    cross_attention_example()
    causal_attention_example()
    transformer_example()
    attention_visualization()
    
    print("\n" + "=" * 60)
    print("示例运行完成！")
    print("\n关键概念总结:")
    print("1. 缩放点积注意力: 计算查询、键、值之间的注意力")
    print("2. 多头注意力: 并行计算多个注意力头，增强表示能力")
    print("3. 自注意力: 序列内部的关系建模")
    print("4. 交叉注意力: 两个序列之间的关系建模")
    print("5. 因果注意力: 用于自回归模型，防止信息泄露")
    print("6. Transformer块: 注意力+前馈网络的完整模块")
    
    print("\n应用场景:")
    print("- 自然语言处理 (BERT, GPT, T5)")
    print("- 机器翻译")
    print("- 文本生成")
    print("- 图像描述生成")
    print("- 语音识别")


if __name__ == "__main__":
    main()