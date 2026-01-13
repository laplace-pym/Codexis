"""
注意力机制实现
包含多种注意力机制：缩放点积注意力、多头注意力等
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ScaledDotProductAttention(nn.Module):
    """缩放点积注意力机制"""
    
    def __init__(self, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query, key, value, mask=None):
        """
        前向传播
        
        Args:
            query: 查询张量 [batch_size, n_heads, seq_len, d_k]
            key: 键张量 [batch_size, n_heads, seq_len, d_k]
            value: 值张量 [batch_size, n_heads, seq_len, d_v]
            mask: 掩码张量 [batch_size, 1, seq_len, seq_len]
            
        Returns:
            注意力输出和注意力权重
        """
        d_k = query.size(-1)
        
        # 计算点积注意力分数
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        
        # 应用掩码（如果有）
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # 应用softmax得到注意力权重
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # 应用注意力权重到值上
        output = torch.matmul(attention_weights, value)
        
        return output, attention_weights


class MultiHeadAttention(nn.Module):
    """多头注意力机制"""
    
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model必须能被n_heads整除"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.d_v = d_model // n_heads
        
        # 线性投影层
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        # 注意力层
        self.attention = ScaledDotProductAttention(dropout)
        
        self.dropout = nn.Dropout(dropout)
        
    def split_heads(self, x):
        """将输入分割成多个头"""
        batch_size, seq_len, _ = x.size()
        x = x.view(batch_size, seq_len, self.n_heads, self.d_k)
        return x.transpose(1, 2)  # [batch_size, n_heads, seq_len, d_k]
    
    def combine_heads(self, x):
        """将多个头合并"""
        batch_size, _, seq_len, _ = x.size()
        x = x.transpose(1, 2).contiguous()
        return x.view(batch_size, seq_len, self.d_model)
    
    def forward(self, query, key, value, mask=None):
        """
        前向传播
        
        Args:
            query: 查询张量 [batch_size, seq_len, d_model]
            key: 键张量 [batch_size, seq_len, d_model]
            value: 值张量 [batch_size, seq_len, d_model]
            mask: 掩码张量 [batch_size, seq_len, seq_len]
            
        Returns:
            多头注意力输出和注意力权重
        """
        batch_size = query.size(0)
        
        # 线性投影
        Q = self.W_q(query)
        K = self.W_k(key)
        V = self.W_v(value)
        
        # 分割成多个头
        Q = self.split_heads(Q)
        K = self.split_heads(K)
        V = self.split_heads(V)
        
        # 调整掩码形状以匹配多头
        if mask is not None:
            mask = mask.unsqueeze(1)  # [batch_size, 1, seq_len, seq_len]
        
        # 应用缩放点积注意力
        attention_output, attention_weights = self.attention(Q, K, V, mask)
        
        # 合并多个头
        output = self.combine_heads(attention_output)
        
        # 最终线性投影
        output = self.W_o(output)
        output = self.dropout(output)
        
        return output, attention_weights


class SelfAttention(nn.Module):
    """自注意力机制"""
    
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.multihead_attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(self, x, mask=None):
        """
        前向传播
        
        Args:
            x: 输入张量 [batch_size, seq_len, d_model]
            mask: 掩码张量 [batch_size, seq_len, seq_len]
            
        Returns:
            自注意力输出和注意力权重
        """
        # 残差连接
        residual = x
        
        # 多头注意力
        output, attention_weights = self.multihead_attention(x, x, x, mask)
        
        # 层归一化和残差连接
        output = self.layer_norm(residual + output)
        
        return output, attention_weights


class CrossAttention(nn.Module):
    """交叉注意力机制"""
    
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.multihead_attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(self, query, key_value, mask=None):
        """
        前向传播
        
        Args:
            query: 查询张量 [batch_size, seq_len_q, d_model]
            key_value: 键值张量 [batch_size, seq_len_kv, d_model]
            mask: 掩码张量 [batch_size, seq_len_q, seq_len_kv]
            
        Returns:
            交叉注意力输出和注意力权重
        """
        # 残差连接
        residual = query
        
        # 多头注意力
        output, attention_weights = self.multihead_attention(query, key_value, key_value, mask)
        
        # 层归一化和残差连接
        output = self.layer_norm(residual + output)
        
        return output, attention_weights


class AdditiveAttention(nn.Module):
    """加性注意力机制（Bahdanau注意力）"""
    
    def __init__(self, hidden_size):
        super().__init__()
        self.W = nn.Linear(hidden_size, hidden_size)
        self.U = nn.Linear(hidden_size, hidden_size)
        self.v = nn.Linear(hidden_size, 1)
        
    def forward(self, query, keys):
        """
        前向传播
        
        Args:
            query: 查询张量 [batch_size, hidden_size]
            keys: 键张量 [batch_size, seq_len, hidden_size]
            
        Returns:
            上下文向量和注意力权重
        """
        # 扩展查询维度以匹配键
        query = query.unsqueeze(1)  # [batch_size, 1, hidden_size]
        
        # 计算注意力分数
        scores = self.v(torch.tanh(self.W(query) + self.U(keys)))  # [batch_size, seq_len, 1]
        scores = scores.squeeze(-1)  # [batch_size, seq_len]
        
        # 应用softmax得到注意力权重
        attention_weights = F.softmax(scores, dim=-1)  # [batch_size, seq_len]
        
        # 计算上下文向量
        context = torch.bmm(attention_weights.unsqueeze(1), keys)  # [batch_size, 1, hidden_size]
        context = context.squeeze(1)  # [batch_size, hidden_size]
        
        return context, attention_weights


class PositionalEncoding(nn.Module):
    """位置编码"""
    
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        
        # 创建位置编码矩阵
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入张量 [batch_size, seq_len, d_model]
            
        Returns:
            添加位置编码后的张量
        """
        return x + self.pe[:, :x.size(1)]


def create_padding_mask(seq, pad_token_id=0):
    """
    创建填充掩码
    
    Args:
        seq: 序列张量 [batch_size, seq_len]
        pad_token_id: 填充标记的ID
        
    Returns:
        填充掩码 [batch_size, 1, 1, seq_len]
    """
    mask = (seq != pad_token_id).unsqueeze(1).unsqueeze(2)
    return mask


def create_look_ahead_mask(size):
    """
    创建前瞻掩码（用于解码器）
    
    Args:
        size: 序列长度
        
    Returns:
        前瞻掩码 [size, size]
    """
    mask = torch.triu(torch.ones(size, size), diagonal=1)
    return mask == 0


if __name__ == "__main__":
    print("测试注意力机制...")
    
    # 测试参数
    batch_size = 2
    seq_len = 10
    d_model = 512
    n_heads = 8
    
    # 创建随机输入
    x = torch.randn(batch_size, seq_len, d_model)
    
    # 测试自注意力
    print(f"\n测试自注意力:")
    print(f"输入形状: {x.shape}")
    
    self_attention = SelfAttention(d_model, n_heads)
    output, attention_weights = self_attention(x)
    
    print(f"输出形状: {output.shape}")
    print(f"注意力权重形状: {attention_weights.shape}")
    
    # 测试多头注意力
    print(f"\n测试多头注意力:")
    multihead_attention = MultiHeadAttention(d_model, n_heads)
    output, attention_weights = multihead_attention(x, x, x)
    
    print(f"输出形状: {output.shape}")
    print(f"注意力权重形状: {attention_weights.shape}")
    
    # 测试加性注意力
    print(f"\n测试加性注意力:")
    hidden_size = 256
    query = torch.randn(batch_size, hidden_size)
    keys = torch.randn(batch_size, seq_len, hidden_size)
    
    additive_attention = AdditiveAttention(hidden_size)
    context, attention_weights = additive_attention(query, keys)
    
    print(f"查询形状: {query.shape}")
    print(f"键形状: {keys.shape}")
    print(f"上下文向量形状: {context.shape}")
    print(f"注意力权重形状: {attention_weights.shape}")
    
    print("\n所有测试完成！")