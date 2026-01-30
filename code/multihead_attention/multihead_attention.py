"""
多头注意力机制 (Multi-Head Attention) 实现
这是Transformer架构中的核心组件
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ScaledDotProductAttention(nn.Module):
    """缩放点积注意力 (Scaled Dot-Product Attention)
    
    这是多头注意力机制中的核心计算单元。
    计算公式: Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V
    """
    
    def __init__(self, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query, key, value, mask=None):
        """
        前向传播
        
        Args:
            query: 查询张量 [batch_size, n_heads, seq_len_q, d_k]
            key: 键张量 [batch_size, n_heads, seq_len_k, d_k]
            value: 值张量 [batch_size, n_heads, seq_len_v, d_v]
            mask: 掩码张量 [batch_size, 1, seq_len_q, seq_len_k] 或 None
            
        Returns:
            output: 注意力输出 [batch_size, n_heads, seq_len_q, d_v]
            attention_weights: 注意力权重 [batch_size, n_heads, seq_len_q, seq_len_k]
        """
        d_k = query.size(-1)  # 获取键的维度
        
        # 1. 计算点积注意力分数: Q * K^T
        # query: [batch_size, n_heads, seq_len_q, d_k]
        # key: [batch_size, n_heads, seq_len_k, d_k]
        # scores: [batch_size, n_heads, seq_len_q, seq_len_k]
        scores = torch.matmul(query, key.transpose(-2, -1))
        
        # 2. 缩放: 除以 sqrt(d_k) 防止梯度消失
        scores = scores / math.sqrt(d_k)
        
        # 3. 应用掩码 (如果有)
        if mask is not None:
            # 将掩码中为0的位置替换为负无穷，这样softmax后权重为0
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # 4. 应用softmax得到注意力权重
        attention_weights = F.softmax(scores, dim=-1)
        
        # 5. 应用dropout防止过拟合
        attention_weights = self.dropout(attention_weights)
        
        # 6. 将注意力权重应用到值上
        output = torch.matmul(attention_weights, value)
        
        return output, attention_weights


class MultiHeadAttention(nn.Module):
    """多头注意力机制 (Multi-Head Attention)
    
    将输入分割成多个头，每个头独立进行注意力计算，
    然后将结果合并。这样可以让模型同时关注不同位置的不同表示子空间。
    """
    
    def __init__(self, d_model, n_heads, dropout=0.1):
        """
        初始化多头注意力
        
        Args:
            d_model: 模型维度 (必须能被n_heads整除)
            n_heads: 注意力头的数量
            dropout: dropout概率
        """
        super().__init__()
        
        # 验证参数
        assert d_model % n_heads == 0, "d_model必须能被n_heads整除"
        
        self.d_model = d_model      # 模型维度
        self.n_heads = n_heads      # 注意力头数量
        self.d_k = d_model // n_heads  # 每个头的键/查询维度
        self.d_v = d_model // n_heads  # 每个头的值维度
        
        # 线性投影层 (将输入投影到查询、键、值空间)
        self.W_q = nn.Linear(d_model, d_model)  # 查询投影
        self.W_k = nn.Linear(d_model, d_model)  # 键投影
        self.W_v = nn.Linear(d_model, d_model)  # 值投影
        self.W_o = nn.Linear(d_model, d_model)  # 输出投影
        
        # 缩放点积注意力层
        self.attention = ScaledDotProductAttention(dropout)
        
        # Dropout层
        self.dropout = nn.Dropout(dropout)
        
    def split_heads(self, x):
        """
        将输入张量分割成多个注意力头
        
        Args:
            x: 输入张量 [batch_size, seq_len, d_model]
            
        Returns:
            分割后的张量 [batch_size, n_heads, seq_len, d_k]
        """
        batch_size, seq_len, _ = x.size()
        
        # 重塑张量: [batch_size, seq_len, n_heads, d_k]
        x = x.view(batch_size, seq_len, self.n_heads, self.d_k)
        
        # 转置: [batch_size, n_heads, seq_len, d_k]
        x = x.transpose(1, 2)
        
        return x
    
    def combine_heads(self, x):
        """
        将多个注意力头的输出合并
        
        Args:
            x: 多头输出 [batch_size, n_heads, seq_len, d_v]
            
        Returns:
            合并后的张量 [batch_size, seq_len, d_model]
        """
        batch_size, n_heads, seq_len, d_v = x.size()
        
        # 转置: [batch_size, seq_len, n_heads, d_v]
        x = x.transpose(1, 2).contiguous()
        
        # 重塑: [batch_size, seq_len, d_model]
        x = x.view(batch_size, seq_len, self.d_model)
        
        return x
    
    def forward(self, query, key, value, mask=None):
        """
        多头注意力前向传播
        
        Args:
            query: 查询张量 [batch_size, seq_len_q, d_model]
            key: 键张量 [batch_size, seq_len_k, d_model]
            value: 值张量 [batch_size, seq_len_v, d_model]
            mask: 掩码张量 [batch_size, seq_len_q, seq_len_k] 或 None
            
        Returns:
            output: 注意力输出 [batch_size, seq_len_q, d_model]
            attention_weights: 注意力权重 [batch_size, n_heads, seq_len_q, seq_len_k]
        """
        batch_size = query.size(0)
        
        # 1. 线性投影: 将输入投影到查询、键、值空间
        Q = self.W_q(query)  # [batch_size, seq_len_q, d_model]
        K = self.W_k(key)    # [batch_size, seq_len_k, d_model]
        V = self.W_v(value)  # [batch_size, seq_len_v, d_model]
        
        # 2. 分割成多个头
        Q = self.split_heads(Q)  # [batch_size, n_heads, seq_len_q, d_k]
        K = self.split_heads(K)  # [batch_size, n_heads, seq_len_k, d_k]
        V = self.split_heads(V)  # [batch_size, n_heads, seq_len_v, d_v]
        
        # 3. 调整掩码形状以匹配多头
        if mask is not None:
            # 添加头维度: [batch_size, 1, seq_len_q, seq_len_k]
            mask = mask.unsqueeze(1)
        
        # 4. 应用缩放点积注意力
        attention_output, attention_weights = self.attention(Q, K, V, mask)
        
        # 5. 合并多个头
        output = self.combine_heads(attention_output)
        
        # 6. 最终线性投影
        output = self.W_o(output)
        output = self.dropout(output)
        
        return output, attention_weights


class PositionalEncoding(nn.Module):
    """位置编码 (Positional Encoding)
    
    为输入序列添加位置信息，因为注意力机制本身没有位置信息。
    使用正弦和余弦函数生成位置编码。
    """
    
    def __init__(self, d_model, max_len=5000):
        """
        初始化位置编码
        
        Args:
            d_model: 模型维度
            max_len: 最大序列长度
        """
        super().__init__()
        
        # 创建位置编码矩阵
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        # 计算除数项
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * 
            (-math.log(10000.0) / d_model)
        )
        
        # 应用正弦函数到偶数位置
        pe[:, 0::2] = torch.sin(position * div_term)
        
        # 应用余弦函数到奇数位置
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # 添加批次维度并注册为缓冲区（不参与训练）
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        """
        为输入添加位置编码
        
        Args:
            x: 输入张量 [batch_size, seq_len, d_model]
            
        Returns:
            添加位置编码后的张量
        """
        return x + self.pe[:, :x.size(1)]


def create_padding_mask(seq, pad_token_id=0):
    """
    创建填充掩码 (Padding Mask)
    
    用于在注意力计算中忽略填充标记
    
    Args:
        seq: 序列张量 [batch_size, seq_len]
        pad_token_id: 填充标记的ID
        
    Returns:
        填充掩码 [batch_size, 1, 1, seq_len]
    """
    # 创建布尔掩码: 非填充位置为True
    mask = (seq != pad_token_id).unsqueeze(1).unsqueeze(2)
    return mask


def create_look_ahead_mask(size):
    """
    创建前瞻掩码 (Look-Ahead Mask)
    
    用于解码器，防止当前位置看到未来的信息
    
    Args:
        size: 序列长度
        
    Returns:
        前瞻掩码 [size, size]
    """
    # 创建上三角矩阵（不包括对角线）
    mask = torch.triu(torch.ones(size, size), diagonal=1)
    
    # 转换为布尔掩码: 允许的位置为True
    return mask == 0


def visualize_attention_weights(attention_weights, title="Attention Weights"):
    """
    可视化注意力权重
    
    Args:
        attention_weights: 注意力权重张量 [batch_size, n_heads, seq_len_q, seq_len_k]
        title: 图表标题
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    # 取第一个批次和第一个头的注意力权重
    weights = attention_weights[0, 0].detach().cpu().numpy()
    
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(weights, cmap='viridis')
    
    ax.set_xlabel('Key Positions')
    ax.set_ylabel('Query Positions')
    ax.set_title(title)
    
    plt.colorbar(im)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    print("=" * 60)
    print("多头注意力机制演示")
    print("=" * 60)
    
    # 设置参数
    batch_size = 2
    seq_len = 8
    d_model = 64
    n_heads = 4
    
    print(f"\n参数设置:")
    print(f"批次大小: {batch_size}")
    print(f"序列长度: {seq_len}")
    print(f"模型维度: {d_model}")
    print(f"注意力头数量: {n_heads}")
    print(f"每个头的维度: d_k = d_v = {d_model // n_heads}")
    
    # 创建随机输入
    print(f"\n创建输入张量...")
    x = torch.randn(batch_size, seq_len, d_model)
    print(f"输入形状: {x.shape}")
    
    # 创建多头注意力层
    print(f"\n创建多头注意力层...")
    multihead_attention = MultiHeadAttention(d_model, n_heads)
    
    # 计算参数数量
    total_params = sum(p.numel() for p in multihead_attention.parameters())
    trainable_params = sum(p.numel() for p in multihead_attention.parameters() if p.requires_grad)
    print(f"总参数数量: {total_params:,}")
    print(f"可训练参数数量: {trainable_params:,}")
    
    # 测试多头注意力
    print(f"\n测试多头注意力...")
    output, attention_weights = multihead_attention(x, x, x)
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")
    print(f"注意力权重形状: {attention_weights.shape}")
    
    # 测试自注意力（查询、键、值相同）
    print(f"\n测试自注意力模式...")
    self_output, self_weights = multihead_attention(x, x, x)
    print(f"自注意力输出形状: {self_output.shape}")
    
    # 测试交叉注意力（查询和键值不同）
    print(f"\n测试交叉注意力模式...")
    query = torch.randn(batch_size, seq_len, d_model)
    key_value = torch.randn(batch_size, seq_len + 2, d_model)
    cross_output, cross_weights = multihead_attention(query, key_value, key_value)
    print(f"查询形状: {query.shape}")
    print(f"键值形状: {key_value.shape}")
    print(f"交叉注意力输出形状: {cross_output.shape}")
    
    # 测试掩码功能
    print(f"\n测试掩码功能...")
    
    # 创建填充掩码
    seq = torch.tensor([[1, 2, 3, 0, 0], [1, 2, 0, 0, 0]])  # 0表示填充
    padding_mask = create_padding_mask(seq)
    print(f"序列: {seq.tolist()}")
    print(f"填充掩码形状: {padding_mask.shape}")
    
    # 创建前瞻掩码
    look_ahead_mask = create_look_ahead_mask(seq_len)
    print(f"前瞻掩码形状: {look_ahead_mask.shape}")
    
    # 测试位置编码
    print(f"\n测试位置编码...")
    positional_encoding = PositionalEncoding(d_model)
    x_with_pe = positional_encoding(x)
    print(f"原始输入形状: {x.shape}")
    print(f"添加位置编码后形状: {x_with_pe.shape}")
    
    print(f"\n" + "=" * 60)
    print("演示完成！")
    print("=" * 60)