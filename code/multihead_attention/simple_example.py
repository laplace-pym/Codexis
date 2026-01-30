"""
多头注意力机制简单示例
展示多头注意力的基本用法和工作原理
"""

import torch
import torch.nn as nn
from multihead_attention import MultiHeadAttention, PositionalEncoding


def simple_attention_demo():
    """简单注意力演示"""
    print("=" * 60)
    print("简单注意力机制演示")
    print("=" * 60)
    
    # 设置简单参数
    batch_size = 1
    seq_len = 4
    d_model = 8
    n_heads = 2
    
    print(f"\n参数:")
    print(f"批次大小: {batch_size}")
    print(f"序列长度: {seq_len}")
    print(f"模型维度: {d_model}")
    print(f"注意力头数量: {n_heads}")
    
    # 创建简单的输入序列
    # 假设我们有4个单词，每个单词用8维向量表示
    input_sequence = torch.tensor([[
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # 单词1
        [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # 单词2
        [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # 单词3
        [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],  # 单词4
    ]])
    
    print(f"\n输入序列形状: {input_sequence.shape}")
    print(f"输入序列 (one-hot编码):")
    for i in range(seq_len):
        print(f"  单词{i+1}: {input_sequence[0, i].numpy()}")
    
    # 创建多头注意力层
    attention = MultiHeadAttention(d_model, n_heads)
    
    # 应用注意力
    output, weights = attention(input_sequence, input_sequence, input_sequence)
    
    print(f"\n输出形状: {output.shape}")
    print(f"注意力权重形状: {weights.shape}")
    
    # 显示注意力权重
    print(f"\n第一个头的注意力权重:")
    head1_weights = weights[0, 0].detach().numpy()
    for i in range(seq_len):
        row = [f"{w:.3f}" for w in head1_weights[i]]
        print(f"  查询位置{i+1}: [{', '.join(row)}]")
    
    return output, weights


def text_attention_example():
    """文本注意力示例"""
    print("\n" + "=" * 60)
    print("文本注意力示例")
    print("=" * 60)
    
    # 模拟一个简单的句子
    sentence = "The cat sat on the mat"
    words = sentence.split()
    seq_len = len(words)
    
    print(f"\n句子: '{sentence}'")
    print(f"单词: {words}")
    print(f"序列长度: {seq_len}")
    
    # 创建词嵌入（简化版）
    d_model = 16
    vocab_size = 10  # 假设有10个不同的单词
    
    # 创建嵌入层
    embedding = nn.Embedding(vocab_size, d_model)
    
    # 为单词分配ID（简化）
    word_ids = torch.tensor([[0, 1, 2, 3, 0, 4]])  # "The"=0, "cat"=1, "sat"=2, "on"=3, "mat"=4
    
    # 获取词嵌入
    embeddings = embedding(word_ids)
    print(f"\n词嵌入形状: {embeddings.shape}")
    
    # 添加位置编码
    pos_encoding = PositionalEncoding(d_model)
    embeddings_with_pos = pos_encoding(embeddings)
    
    # 应用多头注意力
    n_heads = 4
    attention = MultiHeadAttention(d_model, n_heads)
    output, weights = attention(embeddings_with_pos, embeddings_with_pos, embeddings_with_pos)
    
    print(f"注意力输出形状: {output.shape}")
    
    # 分析注意力模式
    print(f"\n分析单词'The'的注意力模式:")
    the_weights = weights[0, :, 0, :].detach().numpy()  # 所有头，查询位置0（第一个"The"）
    
    for head_idx in range(n_heads):
        head_weights = the_weights[head_idx]
        print(f"  头{head_idx+1}: ", end="")
        for word_idx, weight in enumerate(head_weights):
            print(f"{words[word_idx]}:{weight:.3f} ", end="")
        print()
    
    return output, weights


def visualization_example():
    """可视化示例"""
    print("\n" + "=" * 60)
    print("注意力可视化示例")
    print("=" * 60)
    
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        
        # 创建测试数据
        batch_size = 1
        seq_len = 6
        d_model = 12
        n_heads = 3
        
        # 创建输入
        x = torch.randn(batch_size, seq_len, d_model)
        
        # 创建注意力层
        attention = MultiHeadAttention(d_model, n_heads)
        
        # 计算注意力
        output, weights = attention(x, x, x)
        
        # 可视化第一个批次的注意力权重
        fig, axes = plt.subplots(1, n_heads, figsize=(15, 4))
        
        for head_idx in range(n_heads):
            ax = axes[head_idx]
            weight_matrix = weights[0, head_idx].detach().numpy()
            
            im = ax.imshow(weight_matrix, cmap='YlOrRd', vmin=0, vmax=1)
            ax.set_title(f'头 {head_idx+1}')
            ax.set_xlabel('键位置')
            ax.set_ylabel('查询位置')
            
            # 添加网格线
            ax.set_xticks(np.arange(seq_len))
            ax.set_yticks(np.arange(seq_len))
            ax.grid(False)
        
        plt.colorbar(im, ax=axes, orientation='horizontal', fraction=0.05, pad=0.1)
        plt.suptitle('多头注意力权重可视化', fontsize=14)
        plt.tight_layout()
        
        # 保存图像
        plt.savefig('./code/multihead_attention/attention_weights.png', dpi=100, bbox_inches='tight')
        print(f"\n注意力权重图已保存到: ./code/multihead_attention/attention_weights.png")
        
        plt.show()
        
    except ImportError:
        print("Matplotlib未安装，跳过可视化部分")
        print("可以使用以下命令安装: pip install matplotlib")


def practical_application():
    """实际应用示例"""
    print("\n" + "=" * 60)
    print("实际应用示例：句子编码")
    print("=" * 60)
    
    # 模拟两个句子
    sentence1 = "I love machine learning"
    sentence2 = "Deep learning is fascinating"
    
    words1 = sentence1.split()
    words2 = sentence2.split()
    
    print(f"\n句子1: '{sentence1}'")
    print(f"句子2: '{sentence2}'")
    
    # 创建共享的词嵌入
    d_model = 32
    vocab_size = 20
    embedding = nn.Embedding(vocab_size, d_model)
    
    # 为单词分配ID（简化）
    # 假设词汇表: I=0, love=1, machine=2, learning=3, Deep=4, is=5, fascinating=6
    ids1 = torch.tensor([[0, 1, 2, 3]])  # I love machine learning
    ids2 = torch.tensor([[4, 3, 5, 6]])  # Deep learning is fascinating
    
    # 获取词嵌入
    emb1 = embedding(ids1)
    emb2 = embedding(ids2)
    
    # 添加位置编码
    pos_enc = PositionalEncoding(d_model)
    emb1_pos = pos_enc(emb1)
    emb2_pos = pos_enc(emb2)
    
    # 创建多头注意力层
    n_heads = 4
    attention = MultiHeadAttention(d_model, n_heads)
    
    # 自注意力：每个句子内部的关系
    print(f"\n应用自注意力到句子1...")
    output1, weights1 = attention(emb1_pos, emb1_pos, emb1_pos)
    print(f"句子1编码形状: {output1.shape}")
    
    print(f"\n应用自注意力到句子2...")
    output2, weights2 = attention(emb2_pos, emb2_pos, emb2_pos)
    print(f"句子2编码形状: {output2.shape}")
    
    # 交叉注意力：句子1关注句子2
    print(f"\n应用交叉注意力（句子1查询，句子2键值）...")
    cross_output, cross_weights = attention(emb1_pos, emb2_pos, emb2_pos)
    print(f"交叉注意力输出形状: {cross_output.shape}")
    
    # 计算句子相似度（简化）
    sentence1_vector = output1.mean(dim=1)  # 平均池化
    sentence2_vector = output2.mean(dim=1)
    
    similarity = F.cosine_similarity(sentence1_vector, sentence2_vector)
    print(f"\n句子相似度（余弦相似度）: {similarity.item():.4f}")
    
    return output1, output2, cross_output


if __name__ == "__main__":
    print("多头注意力机制示例程序")
    print("=" * 60)
    
    # 运行简单演示
    output1, weights1 = simple_attention_demo()
    
    # 运行文本示例
    output2, weights2 = text_attention_example()
    
    # 运行可视化示例
    visualization_example()
    
    # 运行实际应用示例
    output3, output4, cross_output = practical_application()
    
    print("\n" + "=" * 60)
    print("所有示例完成！")
    print("=" * 60)
    
    print(f"\n总结:")
    print(f"1. 简单演示展示了多头注意力的基本计算")
    print(f"2. 文本示例展示了如何将注意力应用于自然语言")
    print(f"3. 可视化示例帮助理解注意力权重的分布")
    print(f"4. 实际应用示例展示了句子编码和相似度计算")
    
    print(f"\n关键概念:")
    print(f"- 多头注意力允许模型同时关注不同表示子空间")
    print(f"- 每个注意力头学习不同的关注模式")
    print(f"- 位置编码为序列添加位置信息")
    print(f"- 注意力权重显示了输入序列中不同位置之间的关系")